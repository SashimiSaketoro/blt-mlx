#!/usr/bin/env python3
"""
Test script to compare different patcher configurations including the paper's monotonicity settings.
Uses real FineWeb-Edu data from HuggingFace for realistic testing with longer content.
"""

import logging
import torch
import typer
from datatrove.pipeline.readers import ParquetReader
from bytelatent.hf import BltTokenizerAndPatcher
from bytelatent.transformer import LMTransformer
from bytelatent.data.patcher import PatcherArgs, to_device, calculate_entropies

# Configure logging - set to INFO level and output to stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
    force=True  # Force reconfiguration
)
# Also set specific loggers to INFO
logging.getLogger('bytelatent.data.patcher').setLevel(logging.INFO)
logging.getLogger('bytelatent.data.repetition_detector').setLevel(logging.INFO)

app = typer.Typer()

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
    """Extract text from FineWeb-Edu document (handles dict or datatrove objects)."""
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
    num_samples: int = 3,
    limit: int = 1000,
) -> list[str]:
    """Load and concatenate FineWeb-Edu documents to create test samples."""
    print(f"Loading documents from {dataset_path}...")
    print(f"  Target sample size: {target_size} bytes")
    print(f"  Number of samples to create: {num_samples}")
    print(f"  Document limit: {limit}")
    
    try:
        data_reader = ParquetReader(dataset_path, limit=limit)
        
        samples = []
        current_sample = ""
        doc_count = 0
        
        # ParquetReader is callable and returns an iterator
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
        
        print(f"✓ Successfully created {len(samples)} test samples from {doc_count} documents")
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
    patcher_args.threshold_add = config.get('threshold_add')
    patcher_args.monotonicity = config.get('monotonicity', False)
    patcher_args.max_patch_length = config.get('max_patch_length')
    # Repetition detection parameters
    patcher_args.repetition_detection = config.get('repetition_detection', False)
    patcher_args.repetition_window_size = config.get('repetition_window_size', 256)
    patcher_args.repetition_min_match = config.get('repetition_min_match', 8)  # Lower for more sensitivity
    patcher_args.repetition_max_distance = config.get('repetition_max_distance', None)  # None = unlimited
    patcher_args.repetition_hash_size = config.get('repetition_hash_size', 8)  # Should be <= min_match
    
    patcher = patcher_args.build()
    patcher.entropy_model = entropy_model
    return patcher


def main(
    dataset_path: str = typer.Option(
        "hf://datasets/HuggingFaceFW/fineweb-edu/sample/10BT",
        help="HuggingFace dataset path (e.g., hf://datasets/HuggingFaceFW/fineweb-edu/sample/10BT)",
    ),
    sample_size: int = typer.Option(
        50000,
        help="Target size in bytes for concatenated samples",
    ),
    num_samples: int = typer.Option(
        3,
        help="Number of test samples to create",
    ),
    limit: int = typer.Option(
        1000,
        help="Limit total documents to process",
    ),
):
    print("=" * 100)
    print("BLT Patcher Configuration Comparison")
    print("Testing different configurations including paper's monotonicity settings")
    print("=" * 100)
    
    # Check MPS availability
    if torch.backends.mps.is_available():
        device = "mps"
        print(f"✓ MPS is available - using Apple Silicon GPU")
    else:
        device = "cpu"
        print(f"⚠ MPS not available - using CPU")
    
    print(f"\nPyTorch version: {torch.__version__}")
    
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
        
        import os
        os.environ['BLT_SUPPRESS_ATTN_ERROR'] = '1'
        if hasattr(entropy_model, 'attn_impl'):
            entropy_model.attn_impl = 'sdpa'
        
        print("✓ Entropy model loaded successfully")
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
    
    # Define configurations to test
    configs = [
        {
            'name': 'Default HF Model',
            'description': 'Original HuggingFace model configuration',
            'threshold': 1.335442066192627,
            'threshold_add': None,
            'monotonicity': False,
            'max_patch_length': None,
        },
        {
            'name': 'Paper Monotonicity (θ_r=0.35)',
            'description': 'Paper\'s monotonicity constraint with relative threshold',
            'threshold': 0.35,  # Used as relative threshold when monotonicity=True
            'threshold_add': None,
            'monotonicity': True,
            'max_patch_length': None,
        },
        {
            'name': 'Paper Monotonicity + Max Length',
            'description': 'Paper\'s monotonicity with max patch length limit',
            'threshold': 0.35,
            'threshold_add': None,
            'monotonicity': True,
            'max_patch_length': 384,
        },
        {
            'name': 'Combined Global+Relative (θ_g=1.05, θ_r=0.35)',
            'description': 'Combined global and relative thresholds',
            'threshold': 1.05,
            'threshold_add': 0.35,
            'monotonicity': False,
            'max_patch_length': 384,
        },
        {
            'name': 'Combined Higher Threshold (θ_g=1.5, θ_r=0.35)',
            'description': 'Combined with higher global threshold for larger patches',
            'threshold': 1.5,
            'threshold_add': 0.35,
            'monotonicity': False,
            'max_patch_length': 384,
        },
        {
            'name': 'Paper Monotonicity + Repetition Detection',
            'description': 'Paper\'s monotonicity with repetition-aware entropy adjustment',
            'threshold': 0.35,
            'threshold_add': None,
            'monotonicity': True,
            'max_patch_length': 384,
            'repetition_detection': True,
        },
        {
            'name': 'Combined Threshold + Repetition Detection',
            'description': 'Combined thresholds with repetition-aware entropy adjustment',
            'threshold': 1.05,
            'threshold_add': 0.35,
            'monotonicity': False,
            'max_patch_length': 384,
            'repetition_detection': True,
        },
    ]
    
    # Load real FineWeb-Edu data
    print("\n" + "=" * 100)
    print("Loading test samples from FineWeb-Edu...")
    print("=" * 100)
    
    test_samples = create_test_samples_from_fineweb(
        dataset_path=dataset_path,
        target_size=sample_size,
        num_samples=num_samples,
        limit=limit,
    )
    
    # Fallback to synthetic sample if loading fails
    if test_samples is None:
        print("\nUsing synthetic sample as fallback...")
        fineweb_sample = """Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. The field has evolved significantly over the past few decades, with applications ranging from image recognition to natural language processing.

Key Concepts:

1. Supervised Learning: In supervised learning, the algorithm learns from labeled training data. Examples include:
   - Classification: Predicting discrete labels (e.g., spam vs. not spam)
   - Regression: Predicting continuous values (e.g., house prices)

2. Unsupervised Learning: This approach finds hidden patterns in data without labeled examples:
   - Clustering: Grouping similar data points
   - Dimensionality reduction: Reducing the number of features

3. Reinforcement Learning: Agents learn by interacting with an environment and receiving rewards or penalties.

Mathematical Foundations:

The core of machine learning relies on optimization. Given a loss function L(θ) where θ represents model parameters, we seek:

θ* = argmin L(θ)

Common optimization algorithms include gradient descent, stochastic gradient descent, and Adam.

Neural Networks:

Neural networks consist of layers of interconnected nodes (neurons). Each connection has a weight, and each neuron applies an activation function. The forward pass computes:

h = f(Wx + b)

where W is the weight matrix, x is the input, b is the bias, and f is the activation function.

Applications:

Machine learning has revolutionized many fields:
- Computer vision: Object detection, facial recognition
- Natural language processing: Translation, sentiment analysis
- Healthcare: Medical diagnosis, drug discovery
- Finance: Fraud detection, algorithmic trading

Conclusion:

Machine learning continues to advance rapidly, with new architectures and techniques emerging regularly. Understanding the fundamental concepts is crucial for anyone entering this field.

References:
[1] Mitchell, T. M. (1997). Machine Learning. McGraw-Hill.
[2] Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.
[3] Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer.

<div class="footer">
  <p>Copyright © 2024 Educational Content. All rights reserved.</p>
  <p>Last updated: 2024-01-15</p>
  <ul>
    <li><a href="/about">About</a></li>
    <li><a href="/contact">Contact</a></li>
    <li><a href="/privacy">Privacy Policy</a></li>
    <li><a href="/terms">Terms of Service</a></li>
  </ul>
</div>

The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog."""
        test_samples = [fineweb_sample]
    
    print(f"\nLoaded {len(test_samples)} test sample(s):")
    for i, sample in enumerate(test_samples, 1):
        print(f"  Sample {i}: {len(sample)} bytes")
        print(f"    Preview: {sample[:150]}...")
    
    # Test each configuration on all samples
    results = []
    for config in configs:
        print("\n" + "=" * 100)
        print(f"Testing: {config['name']}")
        print(f"Description: {config['description']}")
        print("=" * 100)
        print(f"Configuration:")
        print(f"  - threshold: {config['threshold']}")
        print(f"  - threshold_add: {config.get('threshold_add', 'None')}")
        print(f"  - monotonicity: {config.get('monotonicity', False)}")
        print(f"  - max_patch_length: {config.get('max_patch_length', 'None')}")
        if config.get('repetition_detection', False):
            print(f"  - repetition_detection: True")
            print(f"  - repetition_window_size: {config.get('repetition_window_size', 256)}")
            print(f"  - repetition_min_match: {config.get('repetition_min_match', 8)}")
            max_dist = config.get('repetition_max_distance', None)
            print(f"  - repetition_max_distance: {max_dist if max_dist is not None else 'unlimited'}")
            print(f"  - repetition_hash_size: {config.get('repetition_hash_size', 8)}")
        
        # Test on all samples and aggregate results
        all_stats = []
        processed_samples = []
        for sample_idx, test_sample in enumerate(test_samples, 1):
            try:
                patcher = create_patcher(tok_and_patcher, entropy_model, device, config)
                stats = process_text_to_patches(test_sample, tokenizer, entropy_model, patcher)
                all_stats.append(stats)
                processed_samples.append(test_sample)
                
                if len(test_samples) > 1:
                    print(f"\n  Sample {sample_idx} Results:")
                    print(f"    - Total patches: {stats['num_patches']}")
                    print(f"    - Average patch length: {stats['avg_length']:.1f} bytes")
                    print(f"    - Min/Max patch length: {stats['min_length']}/{stats['max_length']} bytes")
            
            except Exception as e:
                print(f"✗ Error processing sample {sample_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if all_stats:
            # Aggregate statistics across all samples
            total_patches = sum(s['num_patches'] for s in all_stats)
            total_bytes = sum(s['num_patches'] * s['avg_length'] for s in all_stats)
            avg_length = total_bytes / total_patches if total_patches > 0 else 0
            max_length = max(s['max_length'] for s in all_stats)
            total_small = sum(s['small_patches'] for s in all_stats)
            total_medium = sum(s['medium_patches'] for s in all_stats)
            total_large = sum(s['large_patches'] for s in all_stats)
            avg_entropy = sum(s['avg_entropy'] * s['num_patches'] for s in all_stats) / total_patches if total_patches > 0 else 0
            total_input_bytes = sum(len(sample) for sample in processed_samples)
            
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
            
            print(f"\nAggregated Results (across {len(all_stats)} sample(s)):")
            print(f"  - Total patches: {aggregated_stats['num_patches']}")
            print(f"  - Average patch length: {aggregated_stats['avg_length']:.1f} bytes")
            print(f"  - Min/Max patch length: {aggregated_stats['min_length']}/{aggregated_stats['max_length']} bytes")
            print(f"  - Average entropy: {aggregated_stats['avg_entropy']:.4f}")
            print(f"  - Entropy range: {aggregated_stats['entropy_range'][0]:.4f} - {aggregated_stats['entropy_range'][1]:.4f}")
            print(f"  - Compression ratio: {aggregated_stats['compression_ratio']:.2f}x (bytes per patch)")
            print(f"  - Patch size distribution:")
            print(f"    * Small (<10B): {aggregated_stats['small_patches']} ({100*aggregated_stats['small_patches']/aggregated_stats['num_patches']:.1f}%)")
            print(f"    * Medium (10-50B): {aggregated_stats['medium_patches']} ({100*aggregated_stats['medium_patches']/aggregated_stats['num_patches']:.1f}%)")
            print(f"    * Large (≥50B): {aggregated_stats['large_patches']} ({100*aggregated_stats['large_patches']/aggregated_stats['num_patches']:.1f}%)")
    
    # Summary comparison
    print("\n" + "=" * 100)
    print("Configuration Comparison Summary")
    print("=" * 100)
    print(f"{'Configuration':<50} {'Patches':<10} {'Avg Len':<10} {'Max Len':<10} {'Large %':<10}")
    print("-" * 100)
    
    for config, stats in results:
        large_pct = 100 * stats['large_patches'] / stats['num_patches'] if stats['num_patches'] > 0 else 0
        print(f"{config['name']:<50} {stats['num_patches']:<10} {stats['avg_length']:<10.1f} {stats['max_length']:<10} {large_pct:<10.1f}%")
    
    print("\n" + "=" * 100)
    print("✓ All configurations tested!")
    print("=" * 100)


if __name__ == "__main__":
    app.command()(main)
    app()

