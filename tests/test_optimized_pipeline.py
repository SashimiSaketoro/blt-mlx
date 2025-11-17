#!/usr/bin/env python3
"""
Strict monotonicity patching pipeline.
Uses entropy-based patching with strict monotonicity constraint (threshold ≥1.35).
Optimized for large sequences with chunking support.
"""

import logging
import os
import random
import time
from typing import List

import psutil
import torch
import typer
from datasets import load_dataset
from datatrove.pipeline.readers import ParquetReader
from bytelatent.hf import BltTokenizerAndPatcher
from bytelatent.transformer import LMTransformer
from bytelatent.data.patcher import calculate_entropies, PatcherArgs
from bytelatent.data.cross_reference_patcher import strict_monotonicity_patch

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)

app = typer.Typer()


def get_memory_usage_gb():
    """Get current memory usage in GB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)


def get_text_from_doc(doc) -> str:
    """Extract text from datatrove document."""
    if hasattr(doc, 'text'):
        return doc.text
    elif hasattr(doc, 'content'):
        return doc.content
    elif isinstance(doc, dict):
        return doc.get('text', doc.get('content', ''))
    return str(doc)


def create_test_samples_from_fineweb(
    dataset_path: str = "hf://datasets/HuggingFaceFW/fineweb-edu/sample/10BT",
    target_size: int = 50000,
    num_samples: int = 1,
    limit: int = 1000,
) -> list[str]:
    """Load and concatenate FineWeb-Edu documents."""
    try:
        print(f"Loading documents from {dataset_path}...")
        data_reader = ParquetReader(
            data_folder=dataset_path,
            limit=limit,
        )
        
        samples = []
        current_sample = ""
        doc_count = 0
        
        print(f"  Target sample size: {target_size} bytes")
        print(f"  Number of samples to create: {num_samples}")
        
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


def create_test_samples_from_hf_dataset(
    dataset_name: str,
    target_size: int = 50000,
    num_samples: int = 1,
    split: str = "train",
    text_fields: List[str] | None = None,
    seed: int = 42,
) -> list[str]:
    """Sample random rows from a HuggingFace dataset to build test samples."""
    if text_fields is None:
        text_fields = ["text", "content", "instruction", "output", "raw"]

    print(f"Loading dataset '{dataset_name}' (split='{split}') via datasets.load_dataset...")
    dataset = load_dataset(dataset_name, split=split)
    dataset_len = len(dataset)
    print(f"  Loaded {dataset_len} rows from split '{split}'")

    rng = random.Random(seed)
    samples: List[str] = []

    for sample_idx in range(num_samples):
        buffer: list[str] = []
        total_len = 0
        attempts = 0
        max_attempts = max(dataset_len * 5, 1024)

        while total_len < target_size and attempts < max_attempts:
            row = dataset[rng.randint(0, dataset_len - 1)]
            text_segment = ""
            for field in text_fields:
                value = row.get(field)
                if isinstance(value, str) and value.strip():
                    text_segment = value.strip()
                    break
            if not text_segment:
                messages = row.get("messages")
                if isinstance(messages, list):
                    parts = []
                    for message in messages:
                        content = message.get("content")
                        if isinstance(content, str):
                            parts.append(content.strip())
                    text_segment = "\n".join(p for p in parts if p)
            if not text_segment:
                text_segment = str(row)
            if not text_segment:
                attempts += 1
                continue

            buffer.append(text_segment)
            total_len += len(text_segment) + 1
            attempts += 1

        sample_text = "\n\n".join(buffer)
        if len(sample_text) > target_size:
            sample_text = sample_text[:target_size]

        samples.append(sample_text)
        print(f"  Sample {sample_idx + 1}: {len(sample_text)} bytes (assembled from {attempts} segments)")

    return samples


def process_text_to_patches(text: str, tokenizer, entropy_model, device, threshold: float = 1.35):
    """Process text using two-stage cross-reference patching."""
    start_time = time.time()
    
    tokens = tokenizer.encode(text)
    input_ids = torch.tensor([tokens], dtype=torch.long)
    input_ids = input_ids.to(device)
    
    # Create patcher args
    patcher_args = PatcherArgs(
        patching_mode="entropy",
        patching_device=device,
        realtime_patching=False,
        threshold=threshold,
        monotonicity=True,
        max_patch_length=384,
    )
    
    with torch.no_grad():
        # Use strict monotonicity patching
        patch_lengths = strict_monotonicity_patch(
            input_ids,  # Pass 2D tensor [1, seq_len]
            entropy_model,
            patcher_args,
            device=device,
        )
    
    processing_time = time.time() - start_time
    
    # Extract patch lengths (same format as regular patcher)
    # Format should be [batch_size, num_patches, 1] or [batch_size, num_patches]
    if isinstance(patch_lengths, tuple):
        patch_lengths = patch_lengths[0]
    
    if patch_lengths.dim() == 3:
        patch_lengths_1d = patch_lengths[0, :, 0]
    elif patch_lengths.dim() == 2:
        patch_lengths_1d = patch_lengths[0]
    else:
        patch_lengths_1d = patch_lengths
    
    # Filter out zeros and convert to numpy
    patch_lengths_1d = patch_lengths_1d[patch_lengths_1d > 0]
    if isinstance(patch_lengths_1d, torch.Tensor):
        patch_lengths_1d = patch_lengths_1d.cpu().numpy()
    
    # Calculate statistics
    patch_starts = [0] + patch_lengths_1d.cumsum().tolist()[:-1]
    patches_tokens = [tokens[start:start+length] for start, length in zip(patch_starts, patch_lengths_1d)]
    
    # More granular patch size categories
    small_patches = sum(1 for l in patch_lengths_1d if l <= 4)
    small_plus_patches = sum(1 for l in patch_lengths_1d if 5 <= l <= 12)
    medium_patches = sum(1 for l in patch_lengths_1d if 13 <= l <= 24)
    medium_plus_patches = sum(1 for l in patch_lengths_1d if 25 <= l <= 48)
    large_patches = sum(1 for l in patch_lengths_1d if 49 <= l <= 127)
    xl_patches = sum(1 for l in patch_lengths_1d if l >= 128)
    
    sorted_lengths = sorted(patch_lengths_1d)
    p50 = sorted_lengths[len(sorted_lengths) // 2] if sorted_lengths else 0
    p75 = sorted_lengths[int(len(sorted_lengths) * 0.75)] if sorted_lengths else 0
    p90 = sorted_lengths[int(len(sorted_lengths) * 0.90)] if sorted_lengths else 0
    p95 = sorted_lengths[int(len(sorted_lengths) * 0.95)] if sorted_lengths else 0
    
    return {
        'num_patches': len(patch_lengths_1d),
        'avg_length': patch_lengths_1d.mean(),
        'min_length': patch_lengths_1d.min(),
        'max_length': patch_lengths_1d.max(),
        'p50_length': p50,
        'p75_length': p75,
        'p90_length': p90,
        'p95_length': p95,
        'small_patches': small_patches,
        'small_plus_patches': small_plus_patches,
        'medium_patches': medium_patches,
        'medium_plus_patches': medium_plus_patches,
        'large_patches': large_patches,
        'xl_patches': xl_patches,
        'compression_ratio': len(text) / len(patch_lengths_1d) if len(patch_lengths_1d) > 0 else 0,
        'processing_time': processing_time,
    }


def main(
    dataset_path: str = typer.Option(
        "hf://datasets/HuggingFaceFW/fineweb-edu/sample/10BT",
        help="FineWeb parquet path (ignored when --dataset-name is provided)",
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
    threshold: float = typer.Option(
        1.35,
        help="Monotonicity threshold for stage 1 (default: 1.35 for strict)",
    ),
    dataset_name: str | None = typer.Option(
        None,
        help="Optional HuggingFace dataset name to load via datasets.load_dataset",
    ),
    dataset_split: str = typer.Option(
        "train",
        help="Split to use when loading --dataset-name",
    ),
):
    print("=" * 100)
    print("STRICT MONOTONICITY PATCHING")
    print("Entropy-based patching with strict monotonicity (threshold ≥1.35)")
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
        if device == "mps":
            entropy_model = entropy_model.to(device)
            entropy_model = entropy_model.half()  # Use float16 for MPS
            # Change attention implementation to sdpa for macOS compatibility
            entropy_model.attn_impl = "sdpa"
        entropy_model.eval()
        print("✓ Entropy model loaded successfully")
        print(f"Memory after loading model: {get_memory_usage_gb():.2f} GB")
    except Exception as e:
        print(f"✗ Failed to load entropy model: {e}")
        return
    
    # Load tokenizer
    print("\n" + "=" * 100)
    print("Loading tokenizer and patcher configuration...")
    print("=" * 100)
    
    try:
        tok_and_patcher = BltTokenizerAndPatcher.from_pretrained("facebook/blt-7b")
        tokenizer = tok_and_patcher.tokenizer_args.build()
        print("✓ Tokenizer loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load tokenizer: {e}")
        return
    
    # Load test samples
    print("\n" + "=" * 100)
    print("Loading test samples...")
    print("=" * 100)
    
    if dataset_name:
        try:
            test_samples = create_test_samples_from_hf_dataset(
                dataset_name=dataset_name,
                target_size=sample_size,
                num_samples=num_samples,
                split=dataset_split,
            )
        except Exception as e:
            print(f"✗ Failed to load dataset '{dataset_name}': {e}")
            return
    else:
        test_samples = create_test_samples_from_fineweb(
            dataset_path=dataset_path,
            target_size=sample_size,
            num_samples=num_samples,
            limit=limit,
        )
    
    if not test_samples:
        print("✗ Failed to create test samples")
        return
    
    print(f"\nLoaded {len(test_samples)} test sample(s):")
    for i, sample in enumerate(test_samples, 1):
        print(f"  Sample {i}: {len(sample)} bytes")
    
    # Process samples
    dataset_label = dataset_name if dataset_name else dataset_path
    print("\n" + "=" * 100)
    print(f"Testing: Strict Monotonicity Patching (threshold={threshold})")
    print(f"Dataset source: {dataset_label}")
    print("=" * 100)
    
    mem_before = get_memory_usage_gb()
    print(f"Memory before test: {mem_before:.2f} GB")
    
    try:
        all_stats = []
        total_time = 0
        
        for sample_idx, test_sample in enumerate(test_samples, 1):
            stats = process_text_to_patches(test_sample, tokenizer, entropy_model, device, threshold)
            all_stats.append(stats)
            total_time += stats['processing_time']
        
        # Aggregate statistics
        if all_stats:
            total_patches = sum(s['num_patches'] for s in all_stats)
            total_bytes = sum(s['num_patches'] * s['avg_length'] for s in all_stats)
            avg_length = total_bytes / total_patches if total_patches > 0 else 0
            
            total_small = sum(s['small_patches'] for s in all_stats)
            total_small_plus = sum(s['small_plus_patches'] for s in all_stats)
            total_medium = sum(s['medium_patches'] for s in all_stats)
            total_medium_plus = sum(s['medium_plus_patches'] for s in all_stats)
            total_large = sum(s['large_patches'] for s in all_stats)
            total_xl = sum(s['xl_patches'] for s in all_stats)
            
            p50 = sum(s['p50_length'] * s['num_patches'] for s in all_stats) / total_patches if total_patches > 0 else 0
            p75 = sum(s['p75_length'] * s['num_patches'] for s in all_stats) / total_patches if total_patches > 0 else 0
            p90 = sum(s['p90_length'] * s['num_patches'] for s in all_stats) / total_patches if total_patches > 0 else 0
            p95 = sum(s['p95_length'] * s['num_patches'] for s in all_stats) / total_patches if total_patches > 0 else 0
            
            mem_after = get_memory_usage_gb()
            mem_used = mem_after - mem_before
            
            print(f"\nResults:")
            print(f"  - Total patches: {total_patches}")
            print(f"  - Average patch length: {avg_length:.1f} bytes")
            print(f"  - Patch length percentiles:")
            print(f"    * P50 (median): {p50:.1f} bytes")
            print(f"    * P75: {p75:.1f} bytes")
            print(f"    * P90: {p90:.1f} bytes")
            print(f"    * P95: {p95:.1f} bytes")
            print(f"  - Patch size distribution:")
            print(f"    * Small (≤4B): {total_small} ({100*total_small/total_patches:.1f}%)")
            print(f"    * Small+ (5-12B): {total_small_plus} ({100*total_small_plus/total_patches:.1f}%)")
            print(f"    * Medium (13-24B): {total_medium} ({100*total_medium/total_patches:.1f}%)")
            print(f"    * Medium+ (25-48B): {total_medium_plus} ({100*total_medium_plus/total_patches:.1f}%)")
            print(f"    * Large (49-127B): {total_large} ({100*total_large/total_patches:.1f}%)")
            print(f"    * XL (≥128B): {total_xl} ({100*total_xl/total_patches:.1f}%)")
            print(f"  - Processing time: {total_time:.2f} seconds")
            print(f"  - Memory used: {mem_used:.2f} GB")
            print(f"  - Memory after: {mem_after:.2f} GB")
    
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 100)
    print("✓ Optimized pipeline test complete!")
    print("=" * 100)


if __name__ == "__main__":
    app.command()(main)
    app()

