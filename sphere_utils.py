"""
Sphere Utils: Clean interface for BLT strict monotonicity patching.

This module provides simple functions to patch text and datasets using the
strict monotonicity patching pipeline. Designed for use as a dependency in
other projects.

Example usage:
    from sphere_utils import patch_text, patch_dataset
    from datasets import load_dataset

    # Single text
    patches = patch_text("Your text here")

    # HuggingFace Dataset
    ds = load_dataset("RUC-DataLab/DataScience-Instruct-500K", split="train[:100]")
    patches_list = patch_dataset(ds, progress=True)

    # List of strings
    texts = ["text1", "text2", "text3"]
    all_patches = patch_dataset(texts)
"""

import os
from typing import Any, Iterator, Literal, Union

import torch
from datasets import Dataset

from bytelatent.data.cross_reference_patcher import strict_monotonicity_patch
from bytelatent.data.patcher import PatcherArgs
from bytelatent.hf import BltTokenizerAndPatcher
from bytelatent.transformer import LMTransformer

# Set environment variable for macOS compatibility
os.environ.setdefault("BLT_SUPPRESS_ATTN_ERROR", "1")

# Module-level caching for models
_entropy_model_cache: dict[str, Any] = {}
_tokenizer_cache: dict[str, Any] = {}


def _setup_device(device: str | None = None) -> str:
    """
    Auto-detect and setup device (MPS/CPU).

    Args:
        device: Optional device string. If None, auto-detects.

    Returns:
        Device string ("mps", "cpu", or "cuda")
    """
    if device is not None:
        return device

    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def _load_entropy_model(
    entropy_model_path: str = "facebook/blt-entropy",
    device: str | None = None,
) -> torch.nn.Module:
    """
    Load and configure entropy model with caching.

    Args:
        entropy_model_path: Path to entropy model (HF repo or local)
        device: Device to load on (auto-detected if None)

    Returns:
        Loaded and configured entropy model
    """
    cache_key = f"{entropy_model_path}:{device}"

    if cache_key in _entropy_model_cache:
        return _entropy_model_cache[cache_key]

    device = _setup_device(device)

    # Load model
    entropy_model = LMTransformer.from_pretrained(entropy_model_path)

    # Configure for device
    if device == "mps":
        entropy_model = entropy_model.to(device)
        entropy_model = entropy_model.half()  # Use float16 for MPS
        entropy_model.attn_impl = "sdpa"  # macOS compatibility
    elif device == "cuda":
        entropy_model = entropy_model.to(device)
        entropy_model = entropy_model.half()
    else:
        entropy_model = entropy_model.to(device)

    entropy_model.eval()

    # Cache the model
    _entropy_model_cache[cache_key] = entropy_model

    return entropy_model


def _load_tokenizer(tokenizer_path: str = "facebook/blt-7b") -> Any:
    """
    Load tokenizer using BltTokenizerAndPatcher pattern with caching.

    Args:
        tokenizer_path: Path to tokenizer config (HF repo or local)

    Returns:
        Built tokenizer instance
    """
    if tokenizer_path in _tokenizer_cache:
        return _tokenizer_cache[tokenizer_path]

    # Load using built-in pattern
    tok_and_patcher = BltTokenizerAndPatcher.from_pretrained(tokenizer_path)
    tokenizer = tok_and_patcher.tokenizer_args.build()

    # Cache the tokenizer
    _tokenizer_cache[tokenizer_path] = tokenizer

    return tokenizer


def _extract_text_from_item(item: Any, text_field: str | None = None) -> str:
    """
    Extract text from various dataset item formats.

    Similar to get_text_from_doc() in tests/test_optimized_pipeline.py.

    Args:
        item: Dataset item (dict, object with attributes, etc.)
        text_field: Optional field name to extract (if None, tries common fields)

    Returns:
        Extracted text string
    """
    # If already a string, return it
    if isinstance(item, str):
        return item

    # If bytes, decode it
    if isinstance(item, bytes):
        return item.decode("utf-8", errors="ignore")

    # If dict, try various fields
    if isinstance(item, dict):
        if text_field:
            value = item.get(text_field)
            if isinstance(value, str) and value.strip():
                return value.strip()

        # Try common field names
        for field in ["text", "content", "instruction", "output", "raw"]:
            value = item.get(field)
            if isinstance(value, str) and value.strip():
                return value.strip()

        # Try messages format (like DataScience-Instruct-500K)
        messages = item.get("messages")
        if isinstance(messages, list):
            parts = []
            for message in messages:
                if isinstance(message, dict):
                    content = message.get("content")
                    if isinstance(content, str):
                        parts.append(content.strip())
            if parts:
                return "\n".join(p for p in parts if p)

        # Fallback: convert to string
        return str(item)

    # If object with attributes
    if hasattr(item, "text"):
        return str(item.text)
    elif hasattr(item, "content"):
        return str(item.content)

    # Final fallback
    return str(item)


def _format_patches(
    patch_lengths: torch.Tensor,
    tokens: list[int],
    text: str,
    return_format: Literal["patches", "lengths", "both", "detailed"],
) -> Union[list[bytes], list[int], dict]:
    """
    Convert patch_lengths tensor to requested format.

    Args:
        patch_lengths: Tensor of patch lengths [batch_size, num_patches, 1] or similar
        tokens: Original token list (used for byte-level extraction)
        text: Original text string
        return_format: Desired output format

    Returns:
        Formatted patches according to return_format
    """
    # Extract patch lengths from tensor
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

    # Calculate patch starts (in token space)
    patch_starts_tokens = [0] + patch_lengths_1d.cumsum().tolist()[:-1]

    # Convert tokens to bytes for extraction
    # The tokenizer encodes text to tokens, so we need to map back
    # For BLT tokenizer, tokens are byte-level, so we can use them directly
    text_bytes = text.encode("utf-8")
    
    # Extract patches using token positions
    patches = []
    for start_token, length_token in zip(patch_starts_tokens, patch_lengths_1d):
        # Token positions map to byte positions for byte-level tokenizers
        # We need to be careful: patch_lengths are in token space, but we want bytes
        # For BLT tokenizer, tokens are bytes, so this should work
        # But to be safe, let's use the actual text bytes
        # Calculate byte start from token start
        # Since tokens are bytes in BLT, we can use them directly
        start_byte = int(start_token)
        end_byte = int(start_byte + length_token)
        if end_byte <= len(text_bytes):
            patch_bytes = text_bytes[start_byte:end_byte]
            patches.append(patch_bytes)

    # Return according to format
    if return_format == "patches":
        return patches
    elif return_format == "lengths":
        return patch_lengths_1d.tolist()
    elif return_format == "both":
        return {"patches": patches, "lengths": patch_lengths_1d.tolist()}
    elif return_format == "detailed":
        # Calculate statistics
        sorted_lengths = sorted(patch_lengths_1d)
        p50 = sorted_lengths[len(sorted_lengths) // 2] if sorted_lengths else 0
        p75 = sorted_lengths[int(len(sorted_lengths) * 0.75)] if sorted_lengths else 0
        p90 = sorted_lengths[int(len(sorted_lengths) * 0.90)] if sorted_lengths else 0
        p95 = sorted_lengths[int(len(sorted_lengths) * 0.95)] if sorted_lengths else 0

        # Patch size categories
        small = sum(1 for l in patch_lengths_1d if l <= 4)
        small_plus = sum(1 for l in patch_lengths_1d if 5 <= l <= 12)
        medium = sum(1 for l in patch_lengths_1d if 13 <= l <= 24)
        medium_plus = sum(1 for l in patch_lengths_1d if 25 <= l <= 48)
        large = sum(1 for l in patch_lengths_1d if 49 <= l <= 127)
        xl = sum(1 for l in patch_lengths_1d if l >= 128)

        return {
            "patches": patches,
            "lengths": patch_lengths_1d.tolist(),
            "statistics": {
                "num_patches": len(patch_lengths_1d),
                "avg_length": float(patch_lengths_1d.mean()),
                "min_length": int(patch_lengths_1d.min()),
                "max_length": int(patch_lengths_1d.max()),
                "p50": float(p50),
                "p75": float(p75),
                "p90": float(p90),
                "p95": float(p95),
                "small": small,
                "small_plus": small_plus,
                "medium": medium,
                "medium_plus": medium_plus,
                "large": large,
                "xl": xl,
            },
        }
    else:
        raise ValueError(f"Unknown return_format: {return_format}")


def patch_text(
    text: str | bytes,
    threshold: float = 1.35,
    max_patch_length: int = 384,
    device: str | None = None,
    entropy_model_path: str = "facebook/blt-entropy",
    tokenizer_path: str = "facebook/blt-7b",
    return_format: Literal["patches", "lengths", "both", "detailed"] = "patches",
) -> Union[list[bytes], list[int], dict]:
    """
    Patch a single text string using strict monotonicity patching.

    Args:
        text: Input text (string or bytes)
        threshold: Monotonicity threshold (default: 1.35)
        max_patch_length: Maximum patch length (default: 384)
        device: Device to use (auto-detected if None)
        entropy_model_path: Path to entropy model (default: "facebook/blt-entropy")
        tokenizer_path: Path to tokenizer config (default: "facebook/blt-7b")
        return_format: Output format - "patches" (list of bytes), "lengths" (list of ints),
                       "both" (dict), or "detailed" (dict with stats)

    Returns:
        Patches in requested format

    Example:
        >>> patches = patch_text("Your text here")
        >>> # Returns: [b'Your ', b'text ', b'here']
    """
    # Convert bytes to string if needed
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="ignore")

    if not text or not text.strip():
        return [] if return_format == "patches" else {"patches": [], "lengths": []}

    # Setup device
    device = _setup_device(device)

    # Load models (with caching)
    entropy_model = _load_entropy_model(entropy_model_path, device)
    tokenizer = _load_tokenizer(tokenizer_path)

    # Encode text
    tokens = tokenizer.encode(text)
    input_ids = torch.tensor([tokens], dtype=torch.long).to(device)

    # Create patcher args
    patcher_args = PatcherArgs(
        patching_mode="entropy",
        patching_device=device,
        realtime_patching=False,
        threshold=threshold,
        monotonicity=True,
        max_patch_length=max_patch_length,
    )

    # Patch using strict monotonicity
    with torch.no_grad():
        patch_lengths = strict_monotonicity_patch(
            input_ids,
            entropy_model,
            patcher_args,
            device=device,
        )

    # Format and return
    return _format_patches(patch_lengths, tokens, text, return_format)


def patch_dataset(
    dataset: Union[list[str], Dataset, Iterator],
    threshold: float = 1.35,
    max_patch_length: int = 384,
    device: str | None = None,
    entropy_model_path: str = "facebook/blt-entropy",
    tokenizer_path: str = "facebook/blt-7b",
    batch_size: int = 1,
    return_format: Literal["patches", "lengths", "both", "detailed"] = "patches",
    progress: bool = True,
    text_field: str | None = None,
) -> list:
    """
    Patch a dataset (list, HuggingFace Dataset, or iterator) using strict monotonicity patching.

    Args:
        dataset: Input dataset - can be list[str], HuggingFace Dataset, or iterator
        threshold: Monotonicity threshold (default: 1.35)
        max_patch_length: Maximum patch length (default: 384)
        device: Device to use (auto-detected if None)
        entropy_model_path: Path to entropy model (default: "facebook/blt-entropy")
        tokenizer_path: Path to tokenizer config (default: "facebook/blt-7b")
        batch_size: Batch size for processing (default: 1)
        return_format: Output format - "patches", "lengths", "both", or "detailed"
        progress: Show progress bar (default: True)
        text_field: Optional field name to extract from dataset items

    Returns:
        List of patch results (one per item in dataset)

    Example:
        >>> from datasets import load_dataset
        >>> ds = load_dataset("RUC-DataLab/DataScience-Instruct-500K", split="train[:100]")
        >>> patches_list = patch_dataset(ds, progress=True)
    """
    # Setup device and load models once
    device = _setup_device(device)
    entropy_model = _load_entropy_model(entropy_model_path, device)
    tokenizer = _load_tokenizer(tokenizer_path)

    # Convert dataset to iterator
    if isinstance(dataset, Dataset):
        dataset_iter = iter(dataset)
    elif isinstance(dataset, list):
        dataset_iter = iter(dataset)
    else:
        dataset_iter = dataset

    # Progress bar
    if progress:
        try:
            from tqdm import tqdm

            # Try to get length for progress bar
            if isinstance(dataset, (list, Dataset)):
                total = len(dataset)
            else:
                total = None
            pbar = tqdm(total=total, desc="Patching dataset")
        except ImportError:
            pbar = None
    else:
        pbar = None

    results = []
    batch = []

    try:
        for item in dataset_iter:
            # Extract text from item
            text = _extract_text_from_item(item, text_field)

            if not text or not text.strip():
                continue

            batch.append((text, item))

            # Process batch when full
            if len(batch) >= batch_size:
                for text_item, _ in batch:
                    result = patch_text(
                        text_item,
                        threshold=threshold,
                        max_patch_length=max_patch_length,
                        device=device,
                        entropy_model_path=entropy_model_path,
                        tokenizer_path=tokenizer_path,
                        return_format=return_format,
                    )
                    results.append(result)
                    if pbar:
                        pbar.update(1)

                batch = []

        # Process remaining items
        for text_item, _ in batch:
            result = patch_text(
                text_item,
                threshold=threshold,
                max_patch_length=max_patch_length,
                device=device,
                entropy_model_path=entropy_model_path,
                tokenizer_path=tokenizer_path,
                return_format=return_format,
            )
            results.append(result)
            if pbar:
                pbar.update(1)

    finally:
        if pbar:
            pbar.close()

    return results

