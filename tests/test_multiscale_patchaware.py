#!/usr/bin/env python3
"""
Focused test script for multi-scale hierarchical and patch-aware repetition detection.
Tests only the new advanced features with increased iterations to see full effect.
"""

import logging
import os
import time
import psutil
import torch
import typer
from datatrove.pipeline.readers import ParquetReader
from bytelatent.hf import BltTokenizerAndPatcher
from bytelatent.transformer import LMTransformer
from bytelatent.data.patcher import PatcherArgs, to_device, calculate_entropies

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
    force=True
)
logging.getLogger('bytelatent.data.patcher').setLevel(logging.INFO)
logging.getLogger('bytelatent.data.repetition_detector').setLevel(logging.INFO)

app = typer.Typer()


def get_memory_usage_gb():
    """Get current memory usage in GB."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 ** 3)


def process_text_to_patches(text: str, tokenizer, entropy_model, patcher):
    """Process text into patches and return statistics with timing."""
    start_time = time.time()
    
    tokens = tokenizer.encode(text)
    input_ids = torch.tensor([tokens], dtype=torch.long)
    device = next(entropy_model.parameters()).device
    input_ids = input_ids.to(device)
    
    with torch.no_grad():
        entropies, _ = calculate_entropies(
            input_ids,
            entropy_model,
            patching_batch_size=1,
            device=device,
        )
        patch_lengths = patcher.patch(
            input_ids,
            include_next_token=False,
            entropies=entropies,
        )
    
    processing_time = time.time() - start_time
    
    patch_lengths = patch_lengths[0][0].cpu().numpy()
    patch_lengths = patch_lengths[patch_lengths > 0]
    
    # Calculate statistics
    patch_starts = [0] + patch_lengths.cumsum().tolist()[:-1]
    patches_tokens = [tokens[start:start+length] for start, length in zip(patch_starts, patch_lengths)]
    
    patches_text = []
    for patch_tokens in patches_tokens:
        patch_text = tokenizer.decode(patch_tokens, cut_at_eos=False)
        patches_text.append(patch_text)
    
    # Calculate average entropy per patch
    patch_entropies = []
    current_pos = 0
    for length in patch_lengths:
        patch_entropy = entropies[0, current_pos:current_pos+length].mean().item()
        patch_entropies.append(patch_entropy)
        current_pos += length
    
    # Statistics
    avg_entropy = sum(patch_entropies) / len(patch_entropies)
    avg_length = sum(patch_lengths) / len(patch_lengths)
    min_length = min(patch_lengths)
    max_length = max(patch_lengths)
    # More granular patch size categories aligned with actual patch sizes (avg ~3.5 bytes)
    small_patches = sum(1 for l in patch_lengths if l <= 4)
    small_plus_patches = sum(1 for l in patch_lengths if 5 <= l <= 12)
    medium_patches = sum(1 for l in patch_lengths if 13 <= l <= 24)
    medium_plus_patches = sum(1 for l in patch_lengths if 25 <= l <= 48)
    large_patches = sum(1 for l in patch_lengths if 49 <= l <= 127)
    xl_patches = sum(1 for l in patch_lengths if l >= 128)
    
    # Calculate patch length percentiles
    sorted_lengths = sorted(patch_lengths)
    p50 = sorted_lengths[len(sorted_lengths) // 2] if sorted_lengths else 0
    p75 = sorted_lengths[int(len(sorted_lengths) * 0.75)] if sorted_lengths else 0
    p90 = sorted_lengths[int(len(sorted_lengths) * 0.90)] if sorted_lengths else 0
    p95 = sorted_lengths[int(len(sorted_lengths) * 0.95)] if sorted_lengths else 0
    
    return {
        'num_patches': len(patches_text),
        'avg_length': avg_length,
        'min_length': min_length,
        'max_length': max_length,
        'p50_length': p50,
        'p75_length': p75,
        'p90_length': p90,
        'p95_length': p95,
        'avg_entropy': avg_entropy,
        'entropy_range': (min(patch_entropies), max(patch_entropies)),
        'small_patches': small_patches,
        'small_plus_patches': small_plus_patches,
        'medium_patches': medium_patches,
        'medium_plus_patches': medium_plus_patches,
        'large_patches': large_patches,
        'xl_patches': xl_patches,
        'compression_ratio': len(text) / len(patches_text) if len(patches_text) > 0 else 0,
        'processing_time': processing_time,
        'patch_lengths': patch_lengths.tolist(),
        'patch_entropies': patch_entropies,
    }


def get_text_from_doc(doc) -> str:
    """Extract text from FineWeb-Edu document."""
    if isinstance(doc, dict):
        return doc.get("text", doc.get("content", ""))
    elif hasattr(doc, "text"):
        return doc.text
    elif hasattr(doc, "content"):
        return doc.content
    else:
        raise ValueError(f"Could not find text in document: {type(doc)}")


def create_test_samples_from_fineweb(
    dataset_path: str = "hf://datasets/HuggingFaceFW/fineweb-edu/sample/10BT",
    target_size: int = 50000,
    num_samples: int = 1,
    limit: int = 1000,
) -> list[str]:
    """Load and concatenate FineWeb-Edu documents to create test samples."""
    print(f"Loading documents from {dataset_path}...")
    print(f"  Target sample size: {target_size} bytes")
    print(f"  Number of samples to create: {num_samples}")
    
    try:
        data_reader = ParquetReader(dataset_path, limit=limit)
        
        samples = []
        current_sample = ""
        doc_count = 0
        
        for doc in data_reader():
            text = get_text_from_doc(doc)
            if not text or len(text.strip()) == 0:
                continue
                
            doc_count += 1
            
            if len(current_sample) + len(text) < target_size:
                current_sample += "\n\n" + text if current_sample else text
            else:
                if current_sample:
                    samples.append(current_sample)
                    print(f"  Created sample {len(samples)}: {len(current_sample)} bytes from {doc_count} documents")
                current_sample = text
                if len(samples) >= num_samples:
                    break
        
        if current_sample and len(samples) < num_samples:
            samples.append(current_sample)
            print(f"  Created sample {len(samples)}: {len(current_sample)} bytes from {doc_count} documents")
        
        if not samples:
            raise ValueError("No valid samples created from dataset")
        
        print(f"✓ Successfully created {len(samples)} test sample(s) from {doc_count} documents")
        return samples
    
    except Exception as e:
        print(f"⚠ Warning: Failed to load from FineWeb-Edu: {e}")
        print("  Falling back to synthetic sample")
        return None


def create_patcher(tok_and_patcher, entropy_model, device, config):
    """Create a patcher with the given configuration."""
    patcher_args = tok_and_patcher.patcher_args.model_copy(deep=True)
    patcher_args.realtime_patching = False
    patcher_args.patching_device = device
    patcher_args.threshold = config['threshold']
    patcher_args.monotonicity = True  # Always use monotonicity
    patcher_args.max_patch_length = config.get('max_patch_length', 384)
    
    # Repetition detection parameters
    patcher_args.repetition_detection = True  # Always enabled
    patcher_args.repetition_window_size = config.get('repetition_window_size', 256)
    patcher_args.repetition_min_match = config.get('repetition_min_match', 8)
    patcher_args.repetition_max_distance = config.get('repetition_max_distance', None)
    patcher_args.repetition_hash_size = config.get('repetition_hash_size', 8)
    patcher_args.repetition_max_pairs = config.get('repetition_max_pairs', None)
    patcher_args.repetition_sort_by_length = config.get('repetition_sort_by_length', True)
    patcher_args.repetition_max_iterations = config.get('repetition_max_iterations', 7)  # Increased for patch-aware
    patcher_args.repetition_convergence_threshold = config.get('repetition_convergence_threshold', 0.01)
    patcher_args.repetition_batch_size = config.get('repetition_batch_size', 300)
    patcher_args.repetition_multi_scale = config.get('repetition_multi_scale', False)
    patcher_args.repetition_scale_levels = config.get('repetition_scale_levels', [8, 32, 128, 512])
    patcher_args.repetition_patch_aware = config.get('repetition_patch_aware', False)
    patcher_args.repetition_num_windows = config.get('repetition_num_windows', 5)
    patcher_args.repetition_boundary_aware = config.get('repetition_boundary_aware', False)
    patcher_args.repetition_boundary_span_before = config.get('repetition_boundary_span_before', 128)  # Updated default from 64
    patcher_args.repetition_boundary_span_after = config.get('repetition_boundary_span_after', 128)  # Updated default from 64
    patcher_args.repetition_boundary_min_match = config.get('repetition_boundary_min_match', 16)
    
    patcher = patcher_args.build()
    patcher.entropy_model = entropy_model
    return patcher


def main(
    dataset_path: str = typer.Option(
        "hf://datasets/HuggingFaceFW/fineweb-edu/sample/10BT",
        help="HuggingFace dataset path",
    ),
    sample_size: int = typer.Option(
        50000,
        help="Target size in bytes for concatenated samples",
    ),
    num_samples: int = typer.Option(
        1,
        help="Number of test samples to create",
    ),
    limit: int = typer.Option(
        1000,
        help="Limit total documents to process",
    ),
):
    print("=" * 100)
    print("BLT Multi-Scale & Patch-Aware Repetition Detection Testing")
    print("Focused test of hierarchical detection features with increased iterations")
    print("=" * 100)
    
    # Check MPS availability
    if torch.backends.mps.is_available():
        device = "mps"
        print(f"✓ MPS is available - using Apple Silicon GPU")
    else:
        device = "cpu"
        print(f"⚠ MPS not available - using CPU")
    
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"Initial memory usage: {get_memory_usage_gb():.2f} GB")
    
    # Load entropy model
    print("\n" + "=" * 100)
    print("Loading entropy model from HuggingFace...")
    print("=" * 100)
    
    try:
        entropy_model = LMTransformer.from_pretrained("facebook/blt-entropy")
        entropy_model = entropy_model.to(device)
        if device == "mps":
            entropy_model = entropy_model.half()
        entropy_model = entropy_model.eval()
        for param in entropy_model.parameters():
            param.requires_grad = False
        
        os.environ['BLT_SUPPRESS_ATTN_ERROR'] = '1'
        if hasattr(entropy_model, 'attn_impl'):
            entropy_model.attn_impl = 'sdpa'
        
        print("✓ Entropy model loaded successfully")
        print(f"Memory after loading model: {get_memory_usage_gb():.2f} GB")
    except Exception as e:
        print(f"✗ Failed to load entropy model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Load tokenizer and patcher config
    print("\n" + "=" * 100)
    print("Loading tokenizer and patcher configuration...")
    print("=" * 100)
    
    try:
        tok_and_patcher = BltTokenizerAndPatcher.from_pretrained("facebook/blt-7b")
        tokenizer = tok_and_patcher.tokenizer_args.build()
        print("✓ Tokenizer loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load tokenizer/patcher config: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Load test samples
    print("\n" + "=" * 100)
    print("Loading test samples from FineWeb-Edu...")
    print("=" * 100)
    
    test_samples = create_test_samples_from_fineweb(
        dataset_path=dataset_path,
        target_size=sample_size,
        num_samples=num_samples,
        limit=limit,
    )
    
    if test_samples is None:
        print("\nUsing synthetic sample as fallback...")
        test_samples = ["The quick brown fox jumps over the lazy dog. " * 500]
    
    print(f"\nLoaded {len(test_samples)} test sample(s):")
    for i, sample in enumerate(test_samples, 1):
        print(f"  Sample {i}: {len(sample)} bytes")
    
    # Define focused parameter grid - only multi-scale and patch-aware configurations
    parameter_grid = [
        # Baseline for comparison (single-scale, no patch-aware)
        {
            'name': 'Baseline (Single-Scale)',
            'threshold': 0.35,
            'repetition_min_match': 8,
            'repetition_window_size': 256,
            'repetition_hash_size': 8,
            'max_patch_length': 384,
            'repetition_max_pairs': 200,
            'repetition_max_iterations': 7,  # Increased to match others
            'repetition_convergence_threshold': 0.01,
            'repetition_batch_size': 200,
            'repetition_multi_scale': False,
            'repetition_patch_aware': False,
        },
        # Multi-Scale (Basic) - All scales
        {
            'name': 'Multi-Scale (All Scales)',
            'threshold': 0.35,
            'repetition_min_match': 8,
            'repetition_window_size': 256,
            'repetition_hash_size': 8,
            'max_patch_length': 384,
            'repetition_max_pairs': 200,
            'repetition_max_iterations': 7,
            'repetition_convergence_threshold': 0.01,
            'repetition_batch_size': 300,
            'repetition_multi_scale': True,
            'repetition_scale_levels': [8, 32, 128, 512],
            'repetition_patch_aware': False,
        },
        # Multi-Scale (Focused) - Medium scales
        {
            'name': 'Multi-Scale (Focused: 32-256B)',
            'threshold': 0.35,
            'repetition_min_match': 8,
            'repetition_window_size': 256,
            'repetition_hash_size': 8,
            'max_patch_length': 384,
            'repetition_max_pairs': 200,
            'repetition_max_iterations': 7,
            'repetition_convergence_threshold': 0.01,
            'repetition_batch_size': 400,
            'repetition_multi_scale': True,
            'repetition_scale_levels': [32, 128, 256],
            'repetition_patch_aware': False,
        },
        # Patch-Aware Only
        {
            'name': 'Patch-Aware Only',
            'threshold': 0.35,
            'repetition_min_match': 8,
            'repetition_window_size': 256,
            'repetition_hash_size': 8,
            'max_patch_length': 384,
            'repetition_max_pairs': 200,
            'repetition_max_iterations': 7,  # Need iterations > 0 for patch-aware
            'repetition_convergence_threshold': 0.01,
            'repetition_batch_size': 300,
            'repetition_multi_scale': False,
            'repetition_patch_aware': True,
            'repetition_num_windows': 5,
        },
        # Multi-Scale + Patch-Aware (Combined)
        {
            'name': 'Multi-Scale + Patch-Aware',
            'threshold': 0.35,
            'repetition_min_match': 8,
            'repetition_window_size': 256,
            'repetition_hash_size': 8,
            'max_patch_length': 384,
            'repetition_max_pairs': 200,
            'repetition_max_iterations': 7,
            'repetition_convergence_threshold': 0.01,
            'repetition_batch_size': 300,
            'repetition_multi_scale': True,
            'repetition_scale_levels': [8, 32, 128, 512],
            'repetition_patch_aware': True,
            'repetition_num_windows': 5,
        },
        # High-Performance (Large Batches)
        {
            'name': 'High-Performance (Batch=500)',
            'threshold': 0.35,
            'repetition_min_match': 8,
            'repetition_window_size': 256,
            'repetition_hash_size': 8,
            'max_patch_length': 384,
            'repetition_max_pairs': 500,
            'repetition_max_iterations': 5,
            'repetition_convergence_threshold': 0.01,
            'repetition_batch_size': 500,  # Large batches for speed
            'repetition_multi_scale': True,
            'repetition_scale_levels': [8, 32, 128],
            'repetition_patch_aware': False,
        },
        # Boundary-Spanning Detection Only
        {
            'name': 'Boundary-Spanning Only',
            'threshold': 0.35,
            'repetition_min_match': 8,
            'repetition_window_size': 256,
            'repetition_hash_size': 8,
            'max_patch_length': 384,
            'repetition_max_pairs': 200,
            'repetition_max_iterations': 7,  # Need iterations > 0 for boundary-aware
            'repetition_convergence_threshold': 0.01,
            'repetition_batch_size': 300,
            'repetition_multi_scale': False,
            'repetition_patch_aware': False,
            'repetition_boundary_aware': True,
            'repetition_boundary_span_before': 128,  # Updated to new default
            'repetition_boundary_span_after': 128,  # Updated to new default
            'repetition_boundary_min_match': 16,
        },
        # Multi-Scale + Patch-Aware + Boundary-Spanning (All Layers)
        {
            'name': 'All Layers Combined',
            'threshold': 0.35,
            'repetition_min_match': 8,
            'repetition_window_size': 256,
            'repetition_hash_size': 8,
            'max_patch_length': 384,
            'repetition_max_pairs': 200,
            'repetition_max_iterations': 7,
            'repetition_convergence_threshold': 0.01,
            'repetition_batch_size': 300,
            'repetition_multi_scale': True,
            'repetition_scale_levels': [8, 32, 128, 512],
            'repetition_patch_aware': True,
            'repetition_num_windows': 5,
            'repetition_boundary_aware': True,
            'repetition_boundary_span_before': 128,  # Updated to new default
            'repetition_boundary_span_after': 128,  # Updated to new default
            'repetition_boundary_min_match': 16,
        },
    ]
    
    # Test each configuration
    results = []
    for config in parameter_grid:
        print("\n" + "=" * 100)
        print(f"Testing: {config['name']}")
        print("=" * 100)
        print(f"Configuration:")
        print(f"  - threshold: {config['threshold']}")
        print(f"  - repetition_max_iterations: {config['repetition_max_iterations']}")
        print(f"  - repetition_batch_size: {config['repetition_batch_size']}")
        if config.get('repetition_multi_scale', False):
            print(f"  - repetition_multi_scale: True")
            print(f"  - repetition_scale_levels: {config.get('repetition_scale_levels', [])}")
        if config.get('repetition_patch_aware', False):
            print(f"  - repetition_patch_aware: True")
            print(f"  - repetition_num_windows: {config.get('repetition_num_windows', 5)}")
        if config.get('repetition_boundary_aware', False):
            print(f"  - repetition_boundary_aware: True")
            print(f"  - repetition_boundary_span_before: {config.get('repetition_boundary_span_before', 64)}")
            print(f"  - repetition_boundary_span_after: {config.get('repetition_boundary_span_after', 64)}")
            print(f"  - repetition_boundary_min_match: {config.get('repetition_boundary_min_match', 16)}")
        
        mem_before = get_memory_usage_gb()
        print(f"\nMemory before test: {mem_before:.2f} GB")
        
        try:
            patcher = create_patcher(tok_and_patcher, entropy_model, device, config)
            
            # Test on all samples
            all_stats = []
            total_time = 0
            for sample_idx, test_sample in enumerate(test_samples, 1):
                stats = process_text_to_patches(test_sample, tokenizer, entropy_model, patcher)
                all_stats.append(stats)
                total_time += stats['processing_time']
            
            # Aggregate statistics
            if all_stats:
                total_patches = sum(s['num_patches'] for s in all_stats)
                total_bytes = sum(s['num_patches'] * s['avg_length'] for s in all_stats)
                avg_length = total_bytes / total_patches if total_patches > 0 else 0
                max_length = max(s['max_length'] for s in all_stats)
                total_small = sum(s['small_patches'] for s in all_stats)
                total_small_plus = sum(s['small_plus_patches'] for s in all_stats)
                total_medium = sum(s['medium_patches'] for s in all_stats)
                total_medium_plus = sum(s['medium_plus_patches'] for s in all_stats)
                total_large = sum(s['large_patches'] for s in all_stats)
                total_xl = sum(s['xl_patches'] for s in all_stats)
                avg_entropy = sum(s['avg_entropy'] * s['num_patches'] for s in all_stats) / total_patches if total_patches > 0 else 0
                total_input_bytes = sum(len(test_samples[i]) for i in range(len(all_stats)))
                
                # Aggregate percentiles (weighted by number of patches)
                p50 = sum(s['p50_length'] * s['num_patches'] for s in all_stats) / total_patches if total_patches > 0 else 0
                p75 = sum(s['p75_length'] * s['num_patches'] for s in all_stats) / total_patches if total_patches > 0 else 0
                p90 = sum(s['p90_length'] * s['num_patches'] for s in all_stats) / total_patches if total_patches > 0 else 0
                p95 = sum(s['p95_length'] * s['num_patches'] for s in all_stats) / total_patches if total_patches > 0 else 0
                
                aggregated_stats = {
                    'num_patches': total_patches,
                    'avg_length': avg_length,
                    'min_length': min(s['min_length'] for s in all_stats),
                    'max_length': max_length,
                    'p50_length': p50,
                    'p75_length': p75,
                    'p90_length': p90,
                    'p95_length': p95,
                    'avg_entropy': avg_entropy,
                    'entropy_range': (
                        min(s['entropy_range'][0] for s in all_stats),
                        max(s['entropy_range'][1] for s in all_stats),
                    ),
                    'small_patches': total_small,
                    'small_plus_patches': total_small_plus,
                    'medium_patches': total_medium,
                    'medium_plus_patches': total_medium_plus,
                    'large_patches': total_large,
                    'xl_patches': total_xl,
                    'compression_ratio': total_input_bytes / total_patches if total_patches > 0 else 0,
                    'processing_time': total_time,
                }
                
                results.append((config, aggregated_stats))
                
                mem_after = get_memory_usage_gb()
                mem_used = mem_after - mem_before
                
                print(f"\nResults:")
                print(f"  - Total patches: {aggregated_stats['num_patches']}")
                print(f"  - Average patch length: {aggregated_stats['avg_length']:.1f} bytes")
                print(f"  - Patch length percentiles:")
                print(f"    * P50 (median): {aggregated_stats['p50_length']:.1f} bytes")
                print(f"    * P75: {aggregated_stats['p75_length']:.1f} bytes")
                print(f"    * P90: {aggregated_stats['p90_length']:.1f} bytes")
                print(f"    * P95: {aggregated_stats['p95_length']:.1f} bytes")
                print(f"  - Min/Max patch length: {aggregated_stats['min_length']}/{aggregated_stats['max_length']} bytes")
                print(f"  - Average entropy: {aggregated_stats['avg_entropy']:.4f}")
                print(f"  - Entropy range: {aggregated_stats['entropy_range'][0]:.4f} - {aggregated_stats['entropy_range'][1]:.4f}")
                print(f"  - Compression ratio: {aggregated_stats['compression_ratio']:.2f}x")
                print(f"  - Patch size distribution:")
                print(f"    * Small (≤4B): {aggregated_stats['small_patches']} ({100*aggregated_stats['small_patches']/aggregated_stats['num_patches']:.1f}%)")
                print(f"    * Small+ (5-12B): {aggregated_stats['small_plus_patches']} ({100*aggregated_stats['small_plus_patches']/aggregated_stats['num_patches']:.1f}%)")
                print(f"    * Medium (13-24B): {aggregated_stats['medium_patches']} ({100*aggregated_stats['medium_patches']/aggregated_stats['num_patches']:.1f}%)")
                print(f"    * Medium+ (25-48B): {aggregated_stats['medium_plus_patches']} ({100*aggregated_stats['medium_plus_patches']/aggregated_stats['num_patches']:.1f}%)")
                print(f"    * Large (49-127B): {aggregated_stats['large_patches']} ({100*aggregated_stats['large_patches']/aggregated_stats['num_patches']:.1f}%)")
                print(f"    * XL (≥128B): {aggregated_stats['xl_patches']} ({100*aggregated_stats['xl_patches']/aggregated_stats['num_patches']:.1f}%)")
                print(f"  - Processing time: {aggregated_stats['processing_time']:.2f} seconds")
                print(f"  - Memory used: {mem_used:.2f} GB")
                print(f"  - Memory after: {mem_after:.2f} GB")
                
                if mem_after > 20:
                    print(f"  ⚠ WARNING: Memory usage is high ({mem_after:.2f} GB / 24 GB)")
        
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
            mem_after = get_memory_usage_gb()
            if mem_after > 20:
                print(f"⚠ Memory usage after error: {mem_after:.2f} GB - consider reducing sample size or batch size")
    
    # Summary comparison
    print("\n" + "=" * 100)
    print("Multi-Scale & Patch-Aware Comparison Summary")
    print("=" * 100)
    print(f"{'Configuration':<45} {'Patches':<10} {'Avg Len':<10} {'P50':<8} {'P95':<8} {'Distribution':<50} {'Time(s)':<10}")
    print("-" * 100)
    
    for config, stats in results:
        # Calculate distribution percentages
        total = stats['num_patches']
        small_pct = 100 * stats['small_patches'] / total if total > 0 else 0
        small_plus_pct = 100 * stats['small_plus_patches'] / total if total > 0 else 0
        medium_pct = 100 * stats['medium_patches'] / total if total > 0 else 0
        medium_plus_pct = 100 * stats['medium_plus_patches'] / total if total > 0 else 0
        large_pct = 100 * stats['large_patches'] / total if total > 0 else 0
        xl_pct = 100 * stats['xl_patches'] / total if total > 0 else 0
        
        print(f"{config['name']:<45} {stats['num_patches']:<10} {stats['avg_length']:<10.1f} "
              f"{stats['p50_length']:<8.1f} {stats['p95_length']:<8.1f} "
              f"S:{small_pct:.0f}% S+:{small_plus_pct:.0f}% M:{medium_pct:.0f}% M+:{medium_plus_pct:.0f}% L:{large_pct:.0f}% XL:{xl_pct:.0f}% "
              f"{stats['processing_time']:<10.2f}")
    
    print("\n" + "=" * 100)
    print("✓ All multi-scale and patch-aware configurations tested!")
    print("=" * 100)


if __name__ == "__main__":
    app.command()(main)
    app()

