#!/usr/bin/env python3
"""
Focused test script for tuning Paper Monotonicity + Repetition Detection parameters.
Tests various parameter combinations while monitoring memory usage.
"""

import logging
import os
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
logging.getLogger('bytelatent.data.patcher').setLevel(logging.INFO)  # Show detailed logging
logging.getLogger('bytelatent.data.repetition_detector').setLevel(logging.INFO)

app = typer.Typer()


def get_memory_usage_gb():
    """Get current memory usage in GB."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 ** 3)  # Convert to GB


def process_text_to_patches(text: str, tokenizer, entropy_model, patcher):
    """Process text into patches and return statistics."""
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
    small_patches = sum(1 for l in patch_lengths if l < 10)
    medium_patches = sum(1 for l in patch_lengths if 10 <= l < 50)
    large_patches = sum(1 for l in patch_lengths if l >= 50)
    
    return {
        'num_patches': len(patches_text),
        'avg_length': avg_length,
        'min_length': min_length,
        'max_length': max_length,
        'avg_entropy': avg_entropy,
        'entropy_range': (min(patch_entropies), max(patch_entropies)),
        'small_patches': small_patches,
        'medium_patches': medium_patches,
        'large_patches': large_patches,
        'compression_ratio': len(text) / len(patches_text) if len(patches_text) > 0 else 0,
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
    target_size: int = 25000,
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
    patcher_args.repetition_max_pairs = config.get('repetition_max_pairs', None)  # Limit for memory management
    patcher_args.repetition_sort_by_length = config.get('repetition_sort_by_length', True)
    patcher_args.repetition_max_iterations = config.get('repetition_max_iterations', 3)  # Recursive passes
    patcher_args.repetition_convergence_threshold = config.get('repetition_convergence_threshold', 0.01)
    patcher_args.repetition_batch_size = config.get('repetition_batch_size', 200)  # Batch size for processing
    patcher_args.repetition_multi_scale = config.get('repetition_multi_scale', False)  # Multi-scale detection
    patcher_args.repetition_scale_levels = config.get('repetition_scale_levels', [8, 32, 128, 512])  # Scale levels
    patcher_args.repetition_patch_aware = config.get('repetition_patch_aware', False)  # Patch-aware detection
    patcher_args.repetition_num_windows = config.get('repetition_num_windows', 5)  # Number of windows around patches
    
    patcher = patcher_args.build()
    patcher.entropy_model = entropy_model
    return patcher


def main(
    dataset_path: str = typer.Option(
        "hf://datasets/HuggingFaceFW/fineweb-edu/sample/10BT",
        help="HuggingFace dataset path",
    ),
    sample_size: int = typer.Option(
        25000,
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
    print("BLT Repetition Detection Parameter Tuning")
    print("Testing Paper Monotonicity + Repetition Detection with various parameters")
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
        test_samples = ["The quick brown fox jumps over the lazy dog. " * 100]
    
    print(f"\nLoaded {len(test_samples)} test sample(s):")
    for i, sample in enumerate(test_samples, 1):
        print(f"  Sample {i}: {len(sample)} bytes")
    
    # Define parameter grid to test
    parameter_grid = [
        # Baseline
        {
            'name': 'Baseline',
            'threshold': 0.35,
            'repetition_min_match': 8,
            'repetition_window_size': 256,
            'repetition_hash_size': 8,
            'max_patch_length': 384,
            'repetition_max_pairs': None,  # Process all
            'repetition_max_iterations': 3,
            'repetition_convergence_threshold': 0.01,
        },
        # Baseline with memory limit (top 200 longest)
        {
            'name': 'Baseline (Top 200)',
            'threshold': 0.35,
            'repetition_min_match': 8,
            'repetition_window_size': 256,
            'repetition_hash_size': 8,
            'max_patch_length': 384,
            'repetition_max_pairs': 200,  # Limit to top 200 for memory
            'repetition_max_iterations': 3,
            'repetition_convergence_threshold': 0.01,
        },
        # More aggressive recursion
        {
            'name': 'Baseline (5 iterations)',
            'threshold': 0.35,
            'repetition_min_match': 8,
            'repetition_window_size': 256,
            'repetition_hash_size': 8,
            'max_patch_length': 384,
            'repetition_max_pairs': 200,
            'repetition_max_iterations': 5,  # More recursive passes
            'repetition_convergence_threshold': 0.005,  # Stricter convergence
        },
        # More sensitive detection
        {
            'name': 'More Sensitive (min_match=4)',
            'threshold': 0.35,
            'repetition_min_match': 4,
            'repetition_window_size': 256,
            'repetition_hash_size': 4,
            'max_patch_length': 384,
            'repetition_max_pairs': 200,  # Limit for memory (more sensitive = more pairs)
            'repetition_max_iterations': 3,
            'repetition_convergence_threshold': 0.01,
        },
        # Less sensitive detection
        {
            'name': 'Less Sensitive (min_match=12)',
            'threshold': 0.35,
            'repetition_min_match': 12,
            'repetition_window_size': 256,
            'repetition_hash_size': 8,
            'max_patch_length': 384,
            'repetition_max_pairs': None,  # Fewer pairs expected
            'repetition_max_iterations': 3,
            'repetition_convergence_threshold': 0.01,
        },
        # Smaller windows
        {
            'name': 'Smaller Windows (128B)',
            'threshold': 0.35,
            'repetition_min_match': 8,
            'repetition_window_size': 128,
            'repetition_hash_size': 8,
            'max_patch_length': 384,
            'repetition_max_pairs': 200,
            'repetition_max_iterations': 3,
            'repetition_convergence_threshold': 0.01,
        },
        # Larger windows
        {
            'name': 'Larger Windows (512B)',
            'threshold': 0.35,
            'repetition_min_match': 8,
            'repetition_window_size': 512,
            'repetition_hash_size': 8,
            'max_patch_length': 384,
            'repetition_max_pairs': 100,  # Larger windows = more memory per pair
            'repetition_max_iterations': 3,
            'repetition_convergence_threshold': 0.01,
        },
        # Lower threshold
        {
            'name': 'Lower Threshold (0.25)',
            'threshold': 0.25,
            'repetition_min_match': 8,
            'repetition_window_size': 256,
            'repetition_hash_size': 8,
            'max_patch_length': 384,
            'repetition_max_pairs': 200,
            'repetition_max_iterations': 3,
            'repetition_convergence_threshold': 0.01,
        },
        # Higher threshold
        {
            'name': 'Higher Threshold (0.45)',
            'threshold': 0.45,
            'repetition_min_match': 8,
            'repetition_window_size': 256,
            'repetition_hash_size': 8,
            'max_patch_length': 384,
            'repetition_max_pairs': 200,
            'repetition_max_iterations': 3,
            'repetition_convergence_threshold': 0.01,
        },
        # Multi-scale detection
        {
            'name': 'Multi-Scale Detection',
            'threshold': 0.35,
            'repetition_min_match': 8,
            'repetition_window_size': 256,
            'repetition_hash_size': 8,
            'max_patch_length': 384,
            'repetition_max_pairs': 200,
            'repetition_max_iterations': 3,
            'repetition_convergence_threshold': 0.01,
            'repetition_batch_size': 300,  # Larger batch for speed
            'repetition_multi_scale': True,
            'repetition_scale_levels': [8, 32, 128, 512],
        },
        # Multi-scale + Patch-aware
        {
            'name': 'Multi-Scale + Patch-Aware',
            'threshold': 0.35,
            'repetition_min_match': 8,
            'repetition_window_size': 256,
            'repetition_hash_size': 8,
            'max_patch_length': 384,
            'repetition_max_pairs': 200,
            'repetition_max_iterations': 5,  # More iterations for patch-aware
            'repetition_convergence_threshold': 0.01,
            'repetition_batch_size': 300,
            'repetition_multi_scale': True,
            'repetition_scale_levels': [8, 32, 128, 512],
            'repetition_patch_aware': True,
            'repetition_num_windows': 5,
        },
        # High-performance configuration
        {
            'name': 'High-Performance (Large Batches)',
            'threshold': 0.35,
            'repetition_min_match': 8,
            'repetition_window_size': 256,
            'repetition_hash_size': 8,
            'max_patch_length': 384,
            'repetition_max_pairs': 500,  # More pairs
            'repetition_max_iterations': 3,
            'repetition_convergence_threshold': 0.01,
            'repetition_batch_size': 500,  # Large batches for speed
            'repetition_multi_scale': True,
            'repetition_scale_levels': [8, 32, 128],
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
        print(f"  - repetition_min_match: {config['repetition_min_match']}")
        print(f"  - repetition_window_size: {config['repetition_window_size']}")
        print(f"  - repetition_hash_size: {config['repetition_hash_size']}")
        print(f"  - max_patch_length: {config['max_patch_length']}")
        if config.get('repetition_max_pairs'):
            print(f"  - repetition_max_pairs: {config['repetition_max_pairs']} (memory limit)")
        print(f"  - repetition_max_iterations: {config.get('repetition_max_iterations', 3)}")
        print(f"  - repetition_convergence_threshold: {config.get('repetition_convergence_threshold', 0.01)}")
        print(f"  - repetition_batch_size: {config.get('repetition_batch_size', 200)}")
        if config.get('repetition_multi_scale', False):
            print(f"  - repetition_multi_scale: True (scales: {config.get('repetition_scale_levels', [8, 32, 128, 512])})")
        if config.get('repetition_patch_aware', False):
            print(f"  - repetition_patch_aware: True (num_windows: {config.get('repetition_num_windows', 5)})")
        
        mem_before = get_memory_usage_gb()
        print(f"\nMemory before test: {mem_before:.2f} GB")
        
        try:
            patcher = create_patcher(tok_and_patcher, entropy_model, device, config)
            
            # Test on all samples
            all_stats = []
            for sample_idx, test_sample in enumerate(test_samples, 1):
                stats = process_text_to_patches(test_sample, tokenizer, entropy_model, patcher)
                all_stats.append(stats)
            
            # Aggregate statistics
            if all_stats:
                total_patches = sum(s['num_patches'] for s in all_stats)
                total_bytes = sum(s['num_patches'] * s['avg_length'] for s in all_stats)
                avg_length = total_bytes / total_patches if total_patches > 0 else 0
                max_length = max(s['max_length'] for s in all_stats)
                total_small = sum(s['small_patches'] for s in all_stats)
                total_medium = sum(s['medium_patches'] for s in all_stats)
                total_large = sum(s['large_patches'] for s in all_stats)
                avg_entropy = sum(s['avg_entropy'] * s['num_patches'] for s in all_stats) / total_patches if total_patches > 0 else 0
                total_input_bytes = sum(len(test_samples[i]) for i in range(len(all_stats)))
                
                aggregated_stats = {
                    'num_patches': total_patches,
                    'avg_length': avg_length,
                    'min_length': min(s['min_length'] for s in all_stats),
                    'max_length': max_length,
                    'avg_entropy': avg_entropy,
                    'entropy_range': (
                        min(s['entropy_range'][0] for s in all_stats),
                        max(s['entropy_range'][1] for s in all_stats),
                    ),
                    'small_patches': total_small,
                    'medium_patches': total_medium,
                    'large_patches': total_large,
                    'compression_ratio': total_input_bytes / total_patches if total_patches > 0 else 0,
                }
                
                results.append((config, aggregated_stats))
                
                mem_after = get_memory_usage_gb()
                mem_used = mem_after - mem_before
                
                print(f"\nResults:")
                print(f"  - Total patches: {aggregated_stats['num_patches']}")
                print(f"  - Average patch length: {aggregated_stats['avg_length']:.1f} bytes")
                print(f"  - Min/Max patch length: {aggregated_stats['min_length']}/{aggregated_stats['max_length']} bytes")
                print(f"  - Average entropy: {aggregated_stats['avg_entropy']:.4f}")
                print(f"  - Entropy range: {aggregated_stats['entropy_range'][0]:.4f} - {aggregated_stats['entropy_range'][1]:.4f}")
                print(f"  - Compression ratio: {aggregated_stats['compression_ratio']:.2f}x")
                print(f"  - Patch size distribution:")
                print(f"    * Small (<10B): {aggregated_stats['small_patches']} ({100*aggregated_stats['small_patches']/aggregated_stats['num_patches']:.1f}%)")
                print(f"    * Medium (10-50B): {aggregated_stats['medium_patches']} ({100*aggregated_stats['medium_patches']/aggregated_stats['num_patches']:.1f}%)")
                print(f"    * Large (≥50B): {aggregated_stats['large_patches']} ({100*aggregated_stats['large_patches']/aggregated_stats['num_patches']:.1f}%)")
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
    print("Parameter Tuning Comparison Summary")
    print("=" * 100)
    print(f"{'Configuration':<40} {'Patches':<10} {'Avg Len':<10} {'Max Len':<10} {'Large %':<10} {'Mem (GB)':<10}")
    print("-" * 100)
    
    for config, stats in results:
        large_pct = 100 * stats['large_patches'] / stats['num_patches'] if stats['num_patches'] > 0 else 0
        print(f"{config['name']:<40} {stats['num_patches']:<10} {stats['avg_length']:<10.1f} {stats['max_length']:<10} {large_pct:<10.1f}%")
    
    print("\n" + "=" * 100)
    print("✓ All parameter combinations tested!")
    print("=" * 100)


if __name__ == "__main__":
    app.command()(main)
    app()

