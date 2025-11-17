#!/usr/bin/env python3
"""
Test script for sphere_utils.py - simulates usage from another library.

Tests patch_text() and patch_dataset() with different monotonicity thresholds.
"""

import time
from datasets import load_dataset

# Import from sphere_utils (as if it were an external dependency)
from sphere_utils import patch_text, patch_dataset


def test_single_text(threshold: float):
    """Test patch_text() with a single text sample."""
    print(f"\n{'='*80}")
    print(f"Testing patch_text() with threshold={threshold}")
    print(f"{'='*80}")
    
    # Sample text with some repetition
    sample_text = """
    The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog.
    Machine learning is a subset of artificial intelligence. Machine learning models learn from data.
    Natural language processing enables computers to understand human language.
    Deep learning uses neural networks with multiple layers. Deep learning has revolutionized AI.
    """
    
    start_time = time.time()
    result = patch_text(
        sample_text,
        threshold=threshold,
        return_format="detailed"
    )
    elapsed = time.time() - start_time
    
    stats = result["statistics"]
    print(f"  Processing time: {elapsed:.2f} seconds")
    print(f"  Total patches: {stats['num_patches']}")
    print(f"  Average patch length: {stats['avg_length']:.2f} bytes")
    print(f"  Min/Max length: {stats['min_length']}/{stats['max_length']} bytes")
    print(f"  Percentiles: P50={stats['p50']:.1f}, P75={stats['p75']:.1f}, P90={stats['p90']:.1f}, P95={stats['p95']:.1f}")
    print(f"  Distribution:")
    print(f"    Small (≤4B): {stats['small']} ({100*stats['small']/stats['num_patches']:.1f}%)")
    print(f"    Small+ (5-12B): {stats['small_plus']} ({100*stats['small_plus']/stats['num_patches']:.1f}%)")
    print(f"    Medium (13-24B): {stats['medium']} ({100*stats['medium']/stats['num_patches']:.1f}%)")
    print(f"    Medium+ (25-48B): {stats['medium_plus']} ({100*stats['medium_plus']/stats['num_patches']:.1f}%)")
    print(f"    Large (49-127B): {stats['large']} ({100*stats['large']/stats['num_patches']:.1f}%)")
    print(f"    XL (≥128B): {stats['xl']} ({100*stats['xl']/stats['num_patches']:.1f}%)")
    
    return stats


def test_dataset(threshold: float, sample_size: int = 50000):
    """Test patch_dataset() with a HuggingFace dataset."""
    print(f"\n{'='*80}")
    print(f"Testing patch_dataset() with threshold={threshold}")
    print(f"{'='*80}")
    
    try:
        # Load a small sample from a dataset
        print(f"  Loading dataset sample ({sample_size} bytes target)...")
        ds = load_dataset("RUC-DataLab/DataScience-Instruct-500K", split="train[:50]")
        
        # Extract text from first few items to build sample
        texts = []
        total_bytes = 0
        for item in ds:
            if "messages" in item and isinstance(item["messages"], list):
                text_parts = []
                for msg in item["messages"]:
                    if isinstance(msg, dict) and "content" in msg:
                        text_parts.append(msg["content"])
                text = "\n".join(text_parts)
            elif "text" in item:
                text = item["text"]
            else:
                continue
            
            if text and len(text.strip()) > 0:
                texts.append(text)
                total_bytes += len(text)
                if total_bytes >= sample_size:
                    break
        
        if not texts:
            print("  ⚠ No valid texts found in dataset")
            return None
        
        print(f"  Loaded {len(texts)} text samples ({total_bytes} bytes total)")
        
        # Test with list of strings
        start_time = time.time()
        results = patch_dataset(
            texts,
            threshold=threshold,
            return_format="detailed",
            progress=True,
            batch_size=1,
        )
        elapsed = time.time() - start_time
        
        # Aggregate statistics
        total_patches = sum(r["statistics"]["num_patches"] for r in results)
        total_bytes_patched = sum(
            r["statistics"]["num_patches"] * r["statistics"]["avg_length"]
            for r in results
        )
        avg_length = total_bytes_patched / total_patches if total_patches > 0 else 0
        
        # Aggregate distribution
        total_small = sum(r["statistics"]["small"] for r in results)
        total_small_plus = sum(r["statistics"]["small_plus"] for r in results)
        total_medium = sum(r["statistics"]["medium"] for r in results)
        total_medium_plus = sum(r["statistics"]["medium_plus"] for r in results)
        total_large = sum(r["statistics"]["large"] for r in results)
        total_xl = sum(r["statistics"]["xl"] for r in results)
        
        print(f"\n  Results:")
        print(f"    Processing time: {elapsed:.2f} seconds")
        print(f"    Total patches: {total_patches}")
        print(f"    Average patch length: {avg_length:.2f} bytes")
        print(f"    Distribution:")
        print(f"      Small (≤4B): {total_small} ({100*total_small/total_patches:.1f}%)")
        print(f"      Small+ (5-12B): {total_small_plus} ({100*total_small_plus/total_patches:.1f}%)")
        print(f"      Medium (13-24B): {total_medium} ({100*total_medium/total_patches:.1f}%)")
        print(f"      Medium+ (25-48B): {total_medium_plus} ({100*total_medium_plus/total_patches:.1f}%)")
        print(f"      Large (49-127B): {total_large} ({100*total_large/total_patches:.1f}%)")
        print(f"      XL (≥128B): {total_xl} ({100*total_xl/total_patches:.1f}%)")
        
        return {
            "total_patches": total_patches,
            "avg_length": avg_length,
            "processing_time": elapsed,
        }
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Run comparison tests with different thresholds."""
    print("="*80)
    print("SPHERE_UTILS TEST - Monotonicity Threshold Comparison")
    print("="*80)
    print("\nThis script tests sphere_utils.py as it would be used by another library.")
    print("Comparing different monotonicity thresholds: 1.30, 1.35, 1.40, 1.45, 1.50, 1.55")
    
    thresholds = [1.30, 1.35, 1.40, 1.45, 1.50, 1.55]
    
    # Test 1: Single text comparison
    print("\n" + "="*80)
    print("TEST 1: Single Text Comparison")
    print("="*80)
    
    single_text_results = {}
    for threshold in thresholds:
        stats = test_single_text(threshold)
        single_text_results[threshold] = stats
    
    # Summary table for single text
    print("\n" + "="*80)
    print("Single Text Summary")
    print("="*80)
    print(f"{'Threshold':<12} {'Patches':<10} {'Avg Length':<12} {'P50':<8} {'P75':<8} {'P90':<8}")
    print("-"*80)
    for threshold in thresholds:
        stats = single_text_results[threshold]
        print(f"{threshold:<12.2f} {stats['num_patches']:<10} {stats['avg_length']:<12.2f} "
              f"{stats['p50']:<8.1f} {stats['p75']:<8.1f} {stats['p90']:<8.1f}")
    
    # Test 2: Dataset comparison
    print("\n" + "="*80)
    print("TEST 2: Dataset Comparison")
    print("="*80)
    
    dataset_results = {}
    for threshold in thresholds:
        result = test_dataset(threshold, sample_size=50000)
        if result:
            dataset_results[threshold] = result
    
    # Summary table for dataset
    if dataset_results:
        print("\n" + "="*80)
        print("Dataset Summary")
        print("="*80)
        print(f"{'Threshold':<12} {'Patches':<10} {'Avg Length':<12} {'Time (s)':<10}")
        print("-"*80)
        for threshold in thresholds:
            if threshold in dataset_results:
                result = dataset_results[threshold]
                print(f"{threshold:<12.2f} {result['total_patches']:<10} "
                      f"{result['avg_length']:<12.2f} {result['processing_time']:<10.2f}")
    
    print("\n" + "="*80)
    print("✓ All tests complete!")
    print("="*80)


if __name__ == "__main__":
    main()

