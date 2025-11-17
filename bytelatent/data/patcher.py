# Copyright (c) Meta Platforms, Inc. and affiliates.
import logging
import math
import os
import time
from collections import defaultdict
from contextlib import nullcontext
from enum import Enum

import torch
from pydantic import BaseModel
from torch.nn import functional as F

from bytelatent.distributed import get_local_rank
from bytelatent.entropy_model import load_entropy_model

# from src.slurm import get_local_rank
from bytelatent.tokenizers.blt_tokenizer import BPE_ID, OFFSET
from bytelatent.tokenizers.constants import BPE_ID, OFFSET
from bytelatent.data.repetition_detector import (
    detect_repetitions,
    detect_repetitions_multi_scale,
    identify_common_patches,
    extract_multi_windows_around_patches,
    detect_boundary_spanning_repetitions,
)

logger = logging.getLogger(__name__)


class PatchingModeEnum(str, Enum):
    entropy = "entropy"
    bpe = "bpe"
    bpe_patcher = "bpe_patcher"
    space = "space"
    static = "static"
    byte = "byte"


class PatcherArgs(BaseModel):
    patching_mode: PatchingModeEnum = PatchingModeEnum.entropy
    patching_device: str = "cuda"
    entropy_model_checkpoint_dir: str | None = None
    realtime_patching: bool = False
    threshold: float = 1.335442066192627
    threshold_add: float | None = None
    max_patch_length: int | None = None
    patch_size: float = 4.5
    patching_batch_size: int = 1
    device: str = "cuda"
    monotonicity: bool = False
    log_time: bool = False
    # Repetition detection parameters
    repetition_detection: bool = False
    repetition_window_size: int = 256  # Bytes around each occurrence
    repetition_min_match: int = 8  # Minimum bytes to consider repetition (lower = more sensitive)
    repetition_max_distance: int = None  # Max distance to search for repetitions (None = unlimited, can search entire corpus)
    repetition_hash_size: int = 8  # Size of n-gram for hashing (should be <= min_match)
    repetition_max_pairs: int = None  # Maximum number of repetition pairs to process (None = all, useful for memory)
    repetition_sort_by_length: bool = True  # Sort repetitions by length before limiting (longest first)
    repetition_max_iterations: int = 3  # Maximum number of recursive passes (0 = single pass, 1+ = recursive)
    repetition_convergence_threshold: float = 0.01  # Minimum entropy change to continue recursion
    repetition_batch_size: int = 200  # Batch size for processing repetition pairs (50-500, higher = faster but more memory)
    repetition_multi_scale: bool = False  # Enable multi-scale hierarchical detection
    repetition_scale_levels: list[int] = [8, 32, 128, 512]  # Scales for multi-scale detection (in bytes)
    repetition_patch_aware: bool = False  # Detect repetitions around common patches
    repetition_num_windows: int = 5  # Number of 32-byte windows to extract around common patches
    repetition_boundary_aware: bool = False  # Enable boundary-spanning repetition detection
    repetition_boundary_span_before: int = 128  # Bytes before boundary to include in cross-boundary sequence (increased from 64)
    repetition_boundary_span_after: int = 128  # Bytes after boundary to include in cross-boundary sequence (increased from 64)
    repetition_boundary_min_match: int = 16  # Minimum length of cross-boundary sequence to consider

    def build(self) -> "Patcher":
        return Patcher(self)


def entropy(scores):
    """
    scores: [bs, seq_len, vocab]
    returns [bs, seq_len]

    Computes the entropy for each token in the batch.
    Note: uses natural log.
    """
    log_probs = F.log_softmax(scores, dim=-1)
    probs = torch.exp(log_probs)
    p_log_p = log_probs * probs
    entropy = -p_log_p.sum(dim=-1)
    return entropy


def calculate_entropies(
    tokens: torch.tensor,
    entropy_model,
    patching_batch_size,
    device: str | None = None,
    enable_grad: bool = False,
):
    """
    tokens: 2D tensor of shape [batch_size, seq_len]
    Return 2D tensor of shape [batch_size, seq_len] with entropies for each token.

    Splits the tokens into chunks of size max_length and calculates entropies for each chunk.
    Entropy model can be executed on cpu or gpu, specify either 'cuda' or 'cpu' in the device argument.
    """

    grad_context = nullcontext() if enable_grad else torch.no_grad()

    with grad_context:
        entropies = []
        preds = []
        max_length = getattr(entropy_model, "max_length", 8192)
        batch_numel = max_length * patching_batch_size
        splits = torch.split(tokens.flatten(), batch_numel)
        for split in splits:
            pad_size = (max_length - (split.numel() % max_length)) % max_length
            pad = torch.zeros(
                pad_size, dtype=split.dtype, device=split.device, requires_grad=False
            )
            split = torch.cat((split, pad), dim=0)
            split = split.reshape(-1, max_length)
            if device is not None:
                split = split.to(device)
            # assert torch.all(split >= 0) and torch.all(split < 260)
            pred = entropy_model(split)
            pred = pred.reshape(-1, pred.shape[-1])[
                : split.numel() - pad_size, :
            ]  # [batch_size * seq_len, vocab]
            preds.append(pred)
            pred_entropies = entropy(pred)
            entropies.append(pred_entropies)

        concat_entropies = torch.cat(entropies, dim=0)
        concat_entropies = concat_entropies.reshape(tokens.shape)
        concat_preds = torch.cat(preds, dim=0)
        concat_preds = concat_preds.reshape(tokens.shape[0], -1)
    return concat_entropies, concat_preds


def reduce_entropy_differences_at_boundaries(
    entropies: torch.Tensor,
    tokens: torch.Tensor,
    boundary_spanning_pairs: list[tuple[int, int, int, int]],
    patch_start_ids: torch.Tensor,
    threshold: float,
    reduction_factor: float = 0.5,
) -> torch.Tensor:
    """
    Directly reduce entropy differences at boundary positions where boundary-spanning
    repetitions are found. This enables merging by making entropies[i+1] - entropies[i] <= threshold.
    
    Args:
        entropies: Entropy tensor [batch_size, seq_len]
        tokens: Token tensor [batch_size, seq_len]
        boundary_spanning_pairs: List of (start1, end1, start2, end2) tuples from boundary-spanning detection
        patch_start_ids: Original patch start positions [batch_size, num_patches]
        threshold: Entropy difference threshold for patch boundaries
        reduction_factor: Factor to reduce entropy differences (0.0-1.0, default: 0.5 = reduce by 50%)
    
    Returns:
        Adjusted entropy tensor with reduced differences at boundary positions
    """
    if not boundary_spanning_pairs:
        logger.debug("No boundary-spanning pairs provided, returning original entropies")
        return entropies
    
    logger.info(f"    Reducing entropy differences at {len(boundary_spanning_pairs)} boundary position(s)")
    
    # Work with first batch item if 2D
    if entropies.dim() == 2:
        entropies_1d = entropies[0]
    else:
        entropies_1d = entropies
    
    if patch_start_ids.dim() == 2:
        patch_start_ids_1d = patch_start_ids[0]
    else:
        patch_start_ids_1d = patch_start_ids
    
    # Convert patch_start_ids to indices if it's a boolean mask
    if patch_start_ids_1d.dtype == torch.bool:
        patch_start_indices = torch.where(patch_start_ids_1d)[0].cpu().numpy()
    else:
        patch_start_indices = patch_start_ids_1d.cpu().numpy()
    
    # Filter and sort patch starts
    seq_len = entropies_1d.shape[0]
    patch_start_indices = sorted([idx for idx in patch_start_indices if 0 <= idx < seq_len])
    
    if len(patch_start_indices) < 2:
        logger.debug("Not enough patch boundaries for entropy difference reduction")
        return entropies
    
    adjusted_entropies = entropies_1d.clone()
    
    # Find boundary positions within boundary-spanning repetition regions
    boundaries_to_reduce = set()
    for start1, end1, start2, end2 in boundary_spanning_pairs:
        # Find boundaries that fall within these repetition regions
        for boundary_pos in patch_start_indices:
            # Check if boundary is within the repetition region
            if (start1 <= boundary_pos < end1) or (start2 <= boundary_pos < end2):
                boundaries_to_reduce.add(boundary_pos)
    
    if not boundaries_to_reduce:
        logger.debug("No boundaries found within boundary-spanning repetition regions")
        return entropies
    
    logger.info(f"      Found {len(boundaries_to_reduce)} boundary position(s) to reduce")
    
    # Reduce entropy differences at these boundaries
    # Strategy: Reduce entropy at position i+1 to lower the difference entropies[i+1] - entropies[i]
    boundaries_sorted = sorted(boundaries_to_reduce)
    for boundary_pos in boundaries_sorted:
        if boundary_pos >= seq_len - 1:
            continue
        
        # Calculate current entropy difference
        entropy_before = adjusted_entropies[boundary_pos - 1].item() if boundary_pos > 0 else adjusted_entropies[boundary_pos].item()
        entropy_at_boundary = adjusted_entropies[boundary_pos].item()
        entropy_after = adjusted_entropies[boundary_pos + 1].item() if boundary_pos < seq_len - 1 else entropy_at_boundary
        
        # Calculate difference (what triggers boundary creation)
        diff_before = entropy_at_boundary - entropy_before if boundary_pos > 0 else threshold + 1
        diff_after = entropy_after - entropy_at_boundary if boundary_pos < seq_len - 1 else threshold + 1
        
        # If difference is above threshold, reduce it
        if diff_after > threshold:
            # Reduce entropy at boundary_pos+1 to bring difference below threshold
            target_entropy = entropy_at_boundary + threshold * (1 - reduction_factor)
            adjusted_entropies[boundary_pos + 1] = min(adjusted_entropies[boundary_pos + 1], target_entropy)
            logger.debug(f"      Reduced entropy at position {boundary_pos + 1}: {entropy_after:.4f} -> {adjusted_entropies[boundary_pos + 1].item():.4f}")
        
        if diff_before > threshold and boundary_pos > 0:
            # Also reduce entropy at boundary_pos to lower difference from previous position
            target_entropy = entropy_before + threshold * (1 - reduction_factor)
            adjusted_entropies[boundary_pos] = min(adjusted_entropies[boundary_pos], target_entropy)
            logger.debug(f"      Reduced entropy at position {boundary_pos}: {entropy_at_boundary:.4f} -> {adjusted_entropies[boundary_pos].item():.4f}")
    
    # Reshape back to original shape
    if entropies.dim() == 2:
        adjusted_entropies = adjusted_entropies.unsqueeze(0)
        if entropies.shape[0] > 1:
            adjusted_entropies = torch.cat([adjusted_entropies, entropies[1:]], dim=0)
    else:
        adjusted_entropies = adjusted_entropies.unsqueeze(0)
    
    return adjusted_entropies


def recalculate_entropies_for_repetitions(
    entropies: torch.Tensor,
    tokens: torch.Tensor,
    repetition_pairs: list[tuple[int, int, int, int]],
    entropy_model,
    window_size: int = 256,
    device: str | None = None,
    batch_size: int = 200,
    max_context_window: int = 512,
) -> torch.Tensor:
    """
    Recalculate entropies for repetitive regions by maximizing use of the context window.
    Adaptively packs multiple occurrences into the available context (e.g., 4x128B, 2x256B).
    
    Args:
        entropies: Original entropy tensor [batch_size, seq_len]
        tokens: Token tensor [batch_size, seq_len]
        repetition_pairs: List of (start1, end1, start2, end2) tuples
        entropy_model: Entropy model to use for recalculation
        window_size: Base size of context window around each occurrence (default: 256)
        device: Device to run entropy model on
        batch_size: Number of repetition pairs to process at once (default: 200)
        max_context_window: Maximum context window size (default: 512 bytes)
    
    Returns:
        Adjusted entropy tensor with lower entropies for repetitive regions
    """
    if not repetition_pairs:
        logger.debug("No repetition pairs provided, returning original entropies")
        return entropies
    
    logger.info(f"    Recalculating entropies for {len(repetition_pairs)} repetition pair(s)")
    logger.info(f"      Max context window: {max_context_window} bytes (adaptive packing)")
    logger.info(f"      Batch size: {batch_size} pairs per batch")
    
    # Count repetition pairs by match length to show strategy distribution
    short_count = sum(1 for s1, e1, s2, e2 in repetition_pairs if (e1 - s1) < 64)
    medium_count = sum(1 for s1, e1, s2, e2 in repetition_pairs if 64 <= (e1 - s1) < 128)
    long_count = sum(1 for s1, e1, s2, e2 in repetition_pairs if (e1 - s1) >= 128)
    logger.info(f"      Repetition length distribution: {short_count} short (<64B, 128B windows), "
                f"{medium_count} medium (64-128B, 256B windows), {long_count} long (≥128B, 256B windows)")
    
    # Work with first batch item if 2D
    tokens_batch_dim = tokens.shape[0] if tokens.dim() == 2 else 1
    if tokens.dim() == 2:
        tokens_1d = tokens[0]
        entropies_1d = entropies[0]
    else:
        tokens_1d = tokens
        entropies_1d = entropies
    
    # Calculate initial entropy statistics
    initial_mean = entropies_1d.mean().item()
    initial_min = entropies_1d.min().item()
    initial_max = entropies_1d.max().item()
    logger.info(f"      Initial entropy: mean={initial_mean:.4f}, min={initial_min:.4f}, max={initial_max:.4f}")
    
    seq_len = tokens_1d.shape[0]
    adjusted_entropies = entropies_1d.clone()
    
    # Process repetition pairs in batches to manage memory
    num_batches = (len(repetition_pairs) + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(repetition_pairs))
        batch_pairs = repetition_pairs[batch_start:batch_end]
        
        logger.debug(f"Processing batch {batch_idx + 1}/{num_batches} ({len(batch_pairs)} pairs)")
        
        # Optional: Log memory usage if available
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            mem_gb = process.memory_info().rss / (1024 ** 3)
            if mem_gb > 20:
                logger.warning(f"Memory usage high: {mem_gb:.2f} GB before batch {batch_idx + 1}")
        except ImportError:
            pass  # psutil not available, skip memory monitoring
        
        # Process repetition pairs in this batch
        # Group pairs by match length to optimize context window usage
        batch_windows = []
        batch_positions = []  # Store original positions for each window
        batch_window_sizes = []  # Store per_window_size for each pair (for entropy mapping)
        
        for start1, end1, start2, end2 in batch_pairs:
            match_length = end1 - start1
            
            # Determine optimal window size based on match length
            # Goal: maximize use of max_context_window (512 bytes)
            # - Short matches (< 64B): use 128B windows, can fit 4 in 512B
            # - Medium matches (64-128B): use 256B windows, can fit 2 in 512B  
            # - Long matches (> 128B): use 256B windows, fit 2 in 512B
            if match_length < 64:
                # Short repetition: use 128-byte windows, can pack 4 occurrences
                per_window_size = max_context_window // 4  # 128 bytes
                num_windows = 4
            elif match_length < 128:
                # Medium repetition: use 256-byte windows, can pack 2 occurrences
                per_window_size = max_context_window // 2  # 256 bytes
                num_windows = 2
            else:
                # Long repetition: use 256-byte windows, pack 2 occurrences
                per_window_size = max_context_window // 2  # 256 bytes
                num_windows = 2
            
            # Extract windows around each occurrence
            windows = []
            positions = []
            
            for start, end in [(start1, end1), (start2, end2)]:
                center = (start + end) // 2
                
                # Calculate window boundaries
                win_start = max(0, center - per_window_size // 2)
                win_end = min(seq_len, center + per_window_size // 2)
                
                # Extract window
                window = tokens_1d[win_start:win_end]
                
                # Pad window to exactly per_window_size if needed
                if window.shape[0] < per_window_size:
                    pad_size = per_window_size - window.shape[0]
                    if win_start == 0:
                        # Pad at the end
                        pad = torch.zeros(pad_size, dtype=window.dtype, device=window.device)
                        window = torch.cat([window, pad])
                    else:
                        # Pad at the beginning
                        pad = torch.zeros(pad_size, dtype=window.dtype, device=window.device)
                        window = torch.cat([pad, window])
                
                # Truncate to per_window_size if needed
                window = window[:per_window_size]
                
                windows.append(window)
                positions.append((win_start, win_end))
            
            # Concatenate windows to fill max_context_window
            # For num_windows > 2, we only have 2 occurrences, so we'll pad
            concatenated = torch.cat(windows)
            if concatenated.shape[0] < max_context_window:
                # Pad to max_context_window
                pad_size = max_context_window - concatenated.shape[0]
                pad = torch.zeros(pad_size, dtype=concatenated.dtype, device=concatenated.device)
                concatenated = torch.cat([concatenated, pad])
            elif concatenated.shape[0] > max_context_window:
                # Truncate to max_context_window
                concatenated = concatenated[:max_context_window]
            
            batch_windows.append(concatenated)
            batch_positions.append((positions[0][0], positions[0][1], positions[1][0], positions[1][1]))
            batch_window_sizes.append(per_window_size)
        
        if not batch_windows:
            continue
        
        # Process all windows in this batch
        # Stack into batch: [num_windows, 512]
        batch_tokens = torch.stack(batch_windows)
        
        logger.debug(f"Processing {len(batch_windows)} window(s) (each {max_context_window}B) through entropy model")
        
        # Move to device if specified (handle MPS on macOS)
        actual_device = device
        if actual_device is not None:
            # Map "cuda" to actual device if needed
            if actual_device == "cuda" and not torch.cuda.is_available():
                # Try MPS if CUDA not available
                if torch.backends.mps.is_available():
                    actual_device = "mps"
                else:
                    actual_device = "cpu"
            batch_tokens = batch_tokens.to(actual_device)
        
        # Calculate entropies for windows
        with torch.no_grad():
            # Reshape to [batch_size, seq_len] format expected by entropy model
            pred = entropy_model(batch_tokens)
            # pred shape: [num_windows, max_context_window, vocab_size]
            recalculated_entropies = entropy(pred)
            # recalculated_entropies shape: [num_windows, max_context_window]
        
        logger.debug(f"Recalculated entropies shape: {recalculated_entropies.shape}")
        
        # Extract and merge adjusted entropies back into main tensor
        # The concatenated sequence adaptively uses max_context_window (512B)
        # Entropies are split based on per_window_size used for each pair
        for idx, ((win_start1, win_end1, win_start2, win_end2), per_window_size) in enumerate(zip(batch_positions, batch_window_sizes)):
            # Get recalculated entropies for this window pair
            window_entropies = recalculated_entropies[idx]  # [max_context_window]
            
            # Split back into windows based on per_window_size
            # For 2 windows: [window1 (per_window_size) + window2 (per_window_size)] + padding
            window1_entropies = window_entropies[:per_window_size]  # First window
            window2_entropies = window_entropies[per_window_size:per_window_size*2]  # Second window
            
            # Map window1 entropies back to original positions
            # We extracted tokens_1d[win_start1:win_end1], padded to per_window_size, then got entropies
            # The actual window length is win_end1 - win_start1
            win_len1 = win_end1 - win_start1
            if win_len1 > 0:
                # Determine padding: if win_start1 == 0, we padded at the end
                # Otherwise, we padded at the beginning
                if win_start1 == 0:
                    # Padded at end: entropies[0:win_len1] correspond to actual positions
                    adjusted_entropies[win_start1:win_end1] = window1_entropies[:win_len1]
                else:
                    # Padded at beginning: need to find where actual window starts in padded sequence
                    # We padded (per_window_size - win_len1) bytes at the beginning
                    pad_size = per_window_size - win_len1
                    # Entropies[pad_size:pad_size+win_len1] correspond to actual positions
                    adjusted_entropies[win_start1:win_end1] = window1_entropies[pad_size:pad_size + win_len1]
            
            # Map window2 entropies back to original positions (same logic)
            win_len2 = win_end2 - win_start2
            if win_len2 > 0:
                if win_start2 == 0:
                    adjusted_entropies[win_start2:win_end2] = window2_entropies[:win_len2]
                else:
                    pad_size = per_window_size - win_len2
                    adjusted_entropies[win_start2:win_end2] = window2_entropies[pad_size:pad_size + win_len2]
        
        # Clear intermediate tensors to free memory (optimized)
        del batch_tokens, pred, recalculated_entropies
        
        # Periodic cache clearing (every 10 batches to reduce overhead)
        if batch_idx % 10 == 0:
            if actual_device == "mps":
                torch.mps.empty_cache()
            elif actual_device == "cuda":
                torch.cuda.empty_cache()
    
    # Move adjusted entropies back to original device
    original_device = entropies.device
    adjusted_entropies = adjusted_entropies.to(original_device)
    
    # Reshape back to original shape - always maintain 2D shape [batch_size, seq_len]
    if tokens_batch_dim > 1:
        adjusted_entropies = adjusted_entropies.unsqueeze(0)
        # If original had multiple batches, we only adjusted the first one
        if entropies.shape[0] > 1:
            adjusted_entropies = torch.cat([adjusted_entropies, entropies[1:]], dim=0)
    else:
        # Even for single batch, maintain 2D shape
        adjusted_entropies = adjusted_entropies.unsqueeze(0)
    
    # Calculate final entropy statistics
    final_mean = adjusted_entropies.mean().item()
    final_min = adjusted_entropies.min().item()
    final_max = adjusted_entropies.max().item()
    entropy_reduction = initial_mean - final_mean
    logger.info(f"      Final entropy: mean={final_mean:.4f}, min={final_min:.4f}, max={final_max:.4f}")
    logger.info(f"      Entropy reduction: {entropy_reduction:.6f} ({(entropy_reduction/initial_mean*100):.2f}% decrease)")
    logger.info(f"    Recalculation complete for all {len(repetition_pairs)} repetition pairs")
    return adjusted_entropies


def scan_monotonicity_between_sequences(
    entropies: torch.Tensor,
    window_size: int = 32,
    stride: int = 16,
    similarity_threshold: float = 0.1,
) -> list[tuple[int, int, float]]:
    """
    Scan for monotonicity relationships between disparate byte sequences.
    Identifies sequences with similar entropy trajectories (monotonicity patterns)
    that should be treated similarly for patching.
    
    Args:
        entropies: Entropy tensor [batch_size, seq_len] or [seq_len]
        window_size: Size of window to extract for comparison (default: 32)
        stride: Stride for sliding window (default: 16)
        similarity_threshold: Maximum difference in monotonicity score to consider similar (default: 0.1)
    
    Returns:
        List of (start1, start2, similarity_score) tuples for sequences with similar monotonicity
    """
    # Work with 1D tensor
    if entropies.dim() == 2:
        entropies_1d = entropies[0]
    else:
        entropies_1d = entropies
    
    seq_len = entropies_1d.shape[0]
    if seq_len < window_size * 2:
        return []
    
    import numpy as np
    entropies_np = entropies_1d.cpu().numpy()
    
    # Extract windows and compute monotonicity scores
    windows = []
    window_starts = []
    
    for start in range(0, seq_len - window_size + 1, stride):
        window = entropies_np[start:start + window_size]
        # Compute monotonicity score: average of differences (positive = increasing, negative = decreasing)
        differences = np.diff(window)
        monotonicity_score = np.mean(differences)  # Average slope
        windows.append((start, monotonicity_score))
        window_starts.append(start)
    
    # Compare all pairs of windows for similar monotonicity
    similar_pairs = []
    for i, (start1, score1) in enumerate(windows):
        for j, (start2, score2) in enumerate(windows[i + 1:], i + 1):
            # Skip if windows are too close (overlapping or adjacent)
            if abs(start2 - start1) < window_size:
                continue
            
            # Check if monotonicity scores are similar
            score_diff = abs(score1 - score2)
            if score_diff < similarity_threshold:
                # Also check if both have same direction (both increasing or both decreasing)
                if (score1 > 0 and score2 > 0) or (score1 < 0 and score2 < 0) or (abs(score1) < 0.01 and abs(score2) < 0.01):
                    similar_pairs.append((start1, start2, score_diff))
    
    # Sort by similarity (most similar first)
    similar_pairs.sort(key=lambda x: x[2])
    
    return similar_pairs


def patch_start_mask_from_entropy_with_monotonicity(entropies, t):
    """
    entropies: [bs, seq_len] torch tensor of entropies
    t: threshold
    returns [bs, seq_len] mask where True indicates the start of a patch
    
    This function only checks consecutive positions. For scanning monotonicity
    between disparate sequences, use scan_monotonicity_between_sequences().
    """
    bs, seq_len = entropies.shape

    if seq_len == 0:
        return entropies > t

    mask = torch.zeros_like(entropies, dtype=torch.bool)
    mask[:, 0] = True

    # Calculate differences between consecutive elements along the sequence length
    differences = entropies[:, 1:] - entropies[:, :-1]

    # Calculate conditions for all elements except the first one in each sequence
    condition = differences > t

    # Update the mask based on the condition
    mask[:, 1:] = condition

    return mask


def patch_start_mask_global_and_monotonicity(entropies, t, t_add=0):
    """
    entropies: [bs, seq_len] torch tensor of entropies
    t: threshold
    returns [bs, seq_len] mask where True indicates the start of a patch
    """
    bs, seq_len = entropies.shape

    if seq_len == 0:
        return entropies > t

    mask = torch.zeros_like(entropies, dtype=torch.bool)
    mask[:, 0] = True

    # Calculate differences between consecutive elements along the sequence length
    differences = entropies[:, 1:] - entropies[:, :-1]

    # Calculate conditions for all elements except the first one in each sequence
    condition = (differences > t_add) & (entropies[:, 1:] > t) & (~mask[:, :-1])

    # Update the mask based on the condition
    mask[:, 1:] = condition

    return mask


def patch_start_ids_from_patch_start_mask(patch_start_mask):
    bs, trunc_seq_len = patch_start_mask.shape
    max_patches = patch_start_mask.sum(dim=1).max()
    if max_patches == 0:
        patch_start_ids = torch.full(
            (bs, trunc_seq_len),
            trunc_seq_len,
            dtype=torch.long,
            device=patch_start_mask.device,
        )
    else:
        patch_ids = (
            torch.arange(trunc_seq_len, device=patch_start_mask.device)
            .unsqueeze(0)
            .repeat(bs, 1)
        )
        extra_patch_ids = torch.full(
            (bs, trunc_seq_len),
            trunc_seq_len,
            dtype=torch.long,
            device=patch_start_mask.device,
        )
        all_patch_ids = torch.cat((patch_ids, extra_patch_ids), dim=1)
        patch_start_mask_padded = torch.cat(
            (patch_start_mask, ~patch_start_mask), dim=1
        )
        patch_start_ids = all_patch_ids[patch_start_mask_padded].reshape(
            bs, trunc_seq_len
        )[:, :max_patches]
    return patch_start_ids


def check_non_zero_after_zero(tensor):
    zero_mask = tensor == 0
    shifted_mask = torch.cat(
        [
            torch.zeros(tensor.shape[0], 1, dtype=torch.bool, device=tensor.device),
            zero_mask[:, :-1],
        ],
        dim=1,
    )
    non_zero_after_zero = (tensor != 0) & shifted_mask
    return non_zero_after_zero.any()


def patch_lengths_from_start_ids(patch_start_ids, seq_len):
    """
    Calculate patch lengths from start ids.
    start ids: ex: [0, 1, 7, 7, 7, 7, 7], it has the start ids of the patches (here 0, 1), and then
        the rest are filled to the seq len.
    seq_len: ex: 7 length of the sequence

    returns the patch lengths:
    [1, 6] for the above example.
    """
    last_ids = torch.full_like(patch_start_ids[:, :1], seq_len - 1)
    patch_end_ids = torch.cat((patch_start_ids[:, 1:] - 1, last_ids), dim=1)
    patch_lengths = patch_end_ids - patch_start_ids + 1
    assert torch.all(patch_lengths >= 0), f"{patch_lengths}"
    assert not check_non_zero_after_zero(patch_lengths), f"{patch_lengths}"
    return patch_lengths


def find_space_patch_start_ids(tokens):
    bs, seq_len = tokens.shape
    tokens_no_offset = tokens - OFFSET
    patch_end_mask = (
        (tokens_no_offset < ord("0"))
        | ((ord("9") < tokens_no_offset) & (tokens_no_offset < ord("A")))
        | ((ord("Z") < tokens_no_offset) & (tokens_no_offset < ord("a")))
        | ((ord("z") < tokens_no_offset) & (tokens_no_offset < 0b1000_0000))
        | (0b1100_0000 <= tokens_no_offset)
    )
    patch_end_mask[:, 1:] &= patch_end_mask[:, :-1].bitwise_not()
    patch_end_mask |= tokens < OFFSET

    patch_start_mask = torch.cat(
        [
            torch.tensor([1, 1], device=tokens.device, dtype=torch.bool)
            .unsqueeze(0)
            .repeat(bs, 1),
            patch_end_mask[:, 1:],
        ],
        dim=1,
    )
    max_patches = patch_start_mask.sum(dim=1).max()

    patch_ids = (
        torch.arange(seq_len + 1, device=tokens.device).unsqueeze(0).repeat(bs, 1)
    )
    extra_patch_ids = torch.full(
        (bs, seq_len + 1), seq_len + 1, dtype=torch.long, device=tokens.device
    )
    all_patch_ids = torch.cat((patch_ids, extra_patch_ids), dim=1)
    patch_start_mask_padded = torch.cat((patch_start_mask, ~patch_start_mask), dim=1)

    patch_start_ids = all_patch_ids[patch_start_mask_padded].reshape(bs, -1)[
        :, :max_patches
    ]
    return patch_start_ids


def to_device(entropy_model, device=None):
    if device == "cuda":
        rank = get_local_rank()
        device = f"cuda:{rank}"
    entropy_model = entropy_model.to(device)
    return entropy_model, device


def model_pred_to_bpe_patching_pred(pred):
    _, indices = torch.max(pred, dim=1)
    return indices == BPE_ID


def apply_bpe_patcher(tokens, bpe_patcher, patching_batch_size, device=None):
    assert tokens.device == torch.device(
        "cpu"
    ), f"{tokens.device} != cpu expects tokens to be on cpu"
    with torch.no_grad():
        bpe_patcher_device, device = to_device(
            bpe_patcher, device
        )  # Get entropy model to right rank device.
        bpe_patching_mask = []
        max_length = getattr(bpe_patcher, "max_length", 8192)
        batch_numel = max_length * patching_batch_size
        splits = torch.split(tokens.flatten(), batch_numel)
        for split in splits:
            pad_size = (max_length - (split.numel() % max_length)) % max_length
            pad = torch.zeros(
                pad_size, dtype=split.dtype, device=split.device, requires_grad=False
            )
            split = torch.cat((split, pad), dim=0)
            split = split.reshape(-1, max_length).to(device)
            assert torch.all(split >= 0) and torch.all(split < 260)
            pred = bpe_patcher_device(split)
            pred_cpu = pred[0].cpu()
            pred_cpu = pred_cpu.reshape(-1, pred_cpu.shape[-1])[
                : split.numel() - pad_size, :
            ]  # [batch_size * seq_len, vocab]
            bpe_patching_pred = model_pred_to_bpe_patching_pred(pred_cpu)
            bpe_patching_mask.append(bpe_patching_pred)
        bpe_patching_mask = torch.cat(bpe_patching_mask, dim=0)
        bpe_patching_mask = bpe_patching_mask.reshape(tokens.shape)
    return bpe_patching_mask


def find_bpe_patcher_patch_start_ids(
    tokens, bpe_patcher, patching_batch_size, device=None, include_next_token=True
):
    bs, seq_len = tokens.shape

    first_ids = (
        torch.tensor([0, 1], dtype=torch.long, device=tokens.device)
        .unsqueeze(0)
        .repeat(bs, 1)
    )
    preds_truncation_len = first_ids.shape[1]
    token_input = tokens[:, 1:] if include_next_token else tokens[:, 1:-1]
    if token_input.shape[1] >= 1:
        patch_start_mask = apply_bpe_patcher(
            token_input, bpe_patcher, patching_batch_size, device
        )
        assert (
            patch_start_mask.shape[1]
            == tokens.shape[1] + include_next_token - preds_truncation_len
        ), f"{patch_start_mask.shape[1]} != {tokens.shape[1] + include_next_token - preds_truncation_len}"
        patch_start_ids = patch_start_ids_from_patch_start_mask(patch_start_mask)
        patch_start_ids = torch.cat(
            (first_ids, patch_start_ids + preds_truncation_len), dim=1
        )
    else:
        patch_start_ids = first_ids
    return patch_start_ids


def find_entropy_patch_start_ids(
    entropies,
    patch_size=None,
    threshold=None,
    threshold_add=None,
    monotonicity=False,
    include_next_token=True,
    max_patch_length=None,
):
    """
    Use entropies to find the start ids of each patch.
    Use patch_size or threshold to figure out the total number of patches to allocate.

    When threshold is not None the number of patches is not constant between
    different sequences, but patches can be identified incrementally rather than
    decided globally using the entire sequence.
    
    Args:
        max_patch_length: If provided, enforce maximum patch length by forcing
            boundaries every max_patch_length positions. This prevents oversized patches.
    """
    bs, seq_len = entropies.shape[:2]

    first_ids = (
        torch.tensor([0, 1], dtype=torch.long, device=entropies.device)
        .unsqueeze(0)
        .repeat(bs, 1)
    )
    preds_truncation_len = first_ids.shape[
        1
    ]  # remove the first preds because they will be start of patches.
    entropies = entropies[:, 1:]
    if threshold is None:
        num_patches = seq_len // patch_size
        patch_start_ids = entropies.topk(num_patches - 2, dim=1).indices
        patch_start_ids = patch_start_ids.sort(dim=1).values
    else:
        # Assumes that there is at least one token going over the threshold
        if monotonicity:
            patch_start_mask = patch_start_mask_from_entropy_with_monotonicity(
                entropies, threshold
            )
        elif threshold_add is not None and threshold is not None:
            patch_start_mask = patch_start_mask_global_and_monotonicity(
                entropies, threshold, threshold_add
            )
        else:
            patch_start_mask = entropies > threshold
        if not include_next_token:
            patch_start_mask = patch_start_mask[:, :-1]
        
        # Enforce max_patch_length constraint by forcing boundaries
        if max_patch_length is not None:
            # Create forced starts every max_patch_length positions
            # This ensures no patch exceeds max_patch_length
            forced_starts = torch.zeros_like(patch_start_mask, dtype=torch.bool)
            # Start forcing boundaries after the first patch (position 0 is already a start)
            for i in range(max_patch_length, seq_len - preds_truncation_len, max_patch_length):
                if i < patch_start_mask.shape[1]:
                    forced_starts[:, i] = True
            # Merge forced starts with entropy-based starts
            patch_start_mask = patch_start_mask | forced_starts
        
        # patch_start_mask[1:] |= tokens[:-1] < OFFSET
        patch_start_ids = patch_start_ids_from_patch_start_mask(patch_start_mask)

    patch_start_ids = torch.cat(
        (first_ids, patch_start_ids + preds_truncation_len), dim=1
    )
    return patch_start_ids


def rightpad(seq, pad_id, max_len):
    return seq + [pad_id] * (max_len - len(seq))


def find_bpe_delim_patch_start_ids(tokens, delim):
    ids = (tokens[:, :-1] == delim).nonzero(as_tuple=False)
    out = [[0, 1] for _ in range(tokens.shape[0])]
    for x, y in ids:
        # start is at delim + 1, delim should be the last element in the patch.
        out[x.item()].append(y.item() + 1)
    max_len = max([len(elt) for elt in out])
    out = [rightpad(elt, tokens.shape[1], max_len) for elt in out]
    patch_start_ids = torch.tensor(out, dtype=tokens.dtype, device=tokens.device)
    return patch_start_ids


def find_lookup_table_start_mask(
    tokens: torch.Tensor, lookup_table: torch.Tensor, include_next_token=True
):
    window_size = lookup_table.ndim
    # Unfold the tensor to get sliding windows
    unfolded = tokens.unfold(1, window_size, 1)
    # Gather indices for each dimension
    indices = [unfolded[..., i] for i in range(window_size)]
    # Access the lookup table using the gathered indices
    result = lookup_table[indices]
    return result


def find_lookup_table_patch_start_ids(
    tokens: torch.Tensor, lookup_table: torch.Tensor, include_next_token=True
):
    bs, seq_len = tokens.shape

    first_ids = (
        torch.tensor([0, 1], dtype=torch.long, device=tokens.device)
        .unsqueeze(0)
        .repeat(bs, 1)
    )
    preds_truncation_len = first_ids.shape[1]
    window_size = lookup_table.ndim
    assert window_size == 2, f"{window_size} != 2"
    # output dimensions: token_input shape - window_size + 1   --> we want first ids + this = tokens shape + 1 if next token otherwise just token shape
    token_input = (
        tokens if include_next_token else tokens[:, : -preds_truncation_len + 1]
    )
    if token_input.shape[1] >= window_size:
        patch_start_mask = find_lookup_table_start_mask(
            token_input, lookup_table, include_next_token
        )
        assert (
            patch_start_mask.shape[1]
            == tokens.shape[1] + include_next_token - preds_truncation_len
        ), f"{patch_start_mask.shape[1]} != {tokens.shape[1] + include_next_token - preds_truncation_len}"
        patch_start_ids = patch_start_ids_from_patch_start_mask(patch_start_mask)
        patch_start_ids = torch.cat(
            (first_ids, patch_start_ids + preds_truncation_len), dim=1
        )
    else:
        patch_start_ids = first_ids
    return patch_start_ids


def split_large_numbers(lst, m):
    new_lst = []
    for i in lst:
        if i > m:
            while i > m:
                new_lst.append(m)
                i -= m
            new_lst.append(i)
        else:
            new_lst.append(i)
    assert sum(new_lst) == sum(lst), f"{sum(new_lst)} != {sum(lst)}"
    return new_lst


class Patcher:
    def __init__(self, patcher_args: PatcherArgs):
        self.patcher_args = patcher_args
        self.patching_mode = patcher_args.patching_mode
        self.realtime_patching = patcher_args.realtime_patching
        if self.realtime_patching:
            assert (
                patcher_args.entropy_model_checkpoint_dir is not None
            ), "Cannot require realtime patching without an entropy model checkpoint"
            maybe_consolidated = os.path.join(
                patcher_args.entropy_model_checkpoint_dir,
                "consolidated/consolidated.pth",
            )
            if os.path.exists(maybe_consolidated):
                state_path = maybe_consolidated
            else:
                state_path = os.path.join(
                    patcher_args.entropy_model_checkpoint_dir, "consolidated.pth"
                )
            entropy_model, _ = load_entropy_model(
                patcher_args.entropy_model_checkpoint_dir,
                state_path,
            )
            entropy_model, _ = to_device(entropy_model, patcher_args.patching_device)
            self.entropy_model = entropy_model
        else:
            self.entropy_model = None
        self.threshold = patcher_args.threshold
        self.threshold_add = patcher_args.threshold_add
        self.max_patch_length = patcher_args.max_patch_length
        self.patch_size = patcher_args.patch_size
        self.patching_batch_size = patcher_args.patching_batch_size
        self.device = patcher_args.device
        self.monotonicity = patcher_args.monotonicity
        self.log_time = patcher_args.log_time
        # Repetition detection parameters
        self.repetition_detection = patcher_args.repetition_detection
        self.repetition_window_size = patcher_args.repetition_window_size
        self.repetition_min_match = patcher_args.repetition_min_match
        self.repetition_max_distance = patcher_args.repetition_max_distance
        self.repetition_hash_size = patcher_args.repetition_hash_size
        self.repetition_max_pairs = patcher_args.repetition_max_pairs
        self.repetition_sort_by_length = patcher_args.repetition_sort_by_length
        self.repetition_max_iterations = patcher_args.repetition_max_iterations
        self.repetition_convergence_threshold = patcher_args.repetition_convergence_threshold
        self.repetition_batch_size = patcher_args.repetition_batch_size
        self.repetition_multi_scale = patcher_args.repetition_multi_scale
        self.repetition_scale_levels = patcher_args.repetition_scale_levels
        self.repetition_patch_aware = patcher_args.repetition_patch_aware
        self.repetition_num_windows = patcher_args.repetition_num_windows
        self.repetition_boundary_aware = patcher_args.repetition_boundary_aware
        self.repetition_boundary_span_before = patcher_args.repetition_boundary_span_before
        self.repetition_boundary_span_after = patcher_args.repetition_boundary_span_after
        self.repetition_boundary_min_match = patcher_args.repetition_boundary_min_match
        if self.log_time:
            self.log = defaultdict(float)

    def patch(
        self,
        tokens: torch.Tensor,
        include_next_token: bool = False,
        preds: torch.Tensor | None = None,
        entropies: torch.Tensor | None = None,
        threshold: float = None,
    ) -> torch.Tensor:
        """
        tokens: 2D tensor of shape [batch_size, seq_len] that needs to be patched
        Returns patch lengths and optionally scores associated with the tokens (i.e. entropies, logprobs etc.)
        -> output tensor: [batch_size, max_num_patches]
            each tensor is processed independently and gets right padded with zeros.

        Patching with the following modes:
        1. patching_mode = None: static patch size
        2. patching_mode = "entropy":
            calculate entropy of each token, allocate patches so that the total
            number of patches is the same as static patching but choose to begin
            patches on tokens where the model is most uncertain (highest entropy).

            When threshold is provided, it uses the threshold to decide when to
            start a new patch.
        3. patching_mode = "space":
            use space like tokens to define the patches.
        4. patching_mode = "bpe":
            use bpe delim tokens to define the patches.

        To correctly patch the last token, it may be necessary to include the next token in the patch
        lengths calculations. This is controlled by the include_next_token argument.
        """
        bs, seq_len = tokens.shape
        seq_len_next_tok = seq_len + 1 if include_next_token else seq_len
        scores = None
        # STATIC
        if self.patching_mode == PatchingModeEnum.static:
            patch_lengths = torch.zeros(
                (bs, math.ceil(seq_len_next_tok / self.patch_size)),
                dtype=tokens.dtype,
                device=tokens.device,
            ).fill_(self.patch_size)
            if seq_len_next_tok % self.patch_size != 0:
                patch_lengths[:, -1] = seq_len_next_tok % self.patch_size
        elif self.patching_mode == PatchingModeEnum.byte:
            patch_lengths = torch.ones(
                (bs, seq_len_next_tok), dtype=tokens.dtype, device=tokens.device
            )
        # ENTROPY
        elif self.patching_mode == PatchingModeEnum.entropy:
            if self.log_time:
                s = time.time()
            if entropies is not None:
                scores = entropies.to(dtype=torch.float32)
            elif preds is not None:
                scores = entropy(preds)
            else:
                start_entropies = time.time()
                scores, _ = calculate_entropies(
                    tokens,
                    self.entropy_model,
                    self.patching_batch_size,
                    self.device,
                )
            if self.log_time:
                self.log["calculate_entropies"] += time.time() - s
                s = time.time()
            
            # Apply repetition detection and entropy adjustment if enabled
            if self.repetition_detection and self.entropy_model is not None:
                if self.log_time:
                    rep_start = time.time()
                
                logger.info("=" * 80)
                logger.info("REPETITION DETECTION ENABLED")
                seq_len = tokens.shape[1] if tokens.dim() == 2 else tokens.shape[0]
                logger.info(f"  Sequence length: {seq_len} tokens ({seq_len} bytes)")
                max_dist_str = "unlimited" if self.repetition_max_distance is None else str(self.repetition_max_distance)
                max_pairs_str = "all" if self.repetition_max_pairs is None else str(self.repetition_max_pairs)
                logger.info(f"  Detection parameters:")
                logger.info(f"    - min_match: {self.repetition_min_match} bytes")
                logger.info(f"    - max_distance: {max_dist_str}")
                logger.info(f"    - window_size: {self.repetition_window_size} bytes")
                logger.info(f"    - hash_size: {self.repetition_hash_size} bytes")
                logger.info(f"    - max_pairs: {max_pairs_str}")
                logger.info(f"    - sort_by_length: {self.repetition_sort_by_length}")
                logger.info(f"  Recursion parameters:")
                logger.info(f"    - max_iterations: {self.repetition_max_iterations}")
                logger.info(f"    - convergence_threshold: {self.repetition_convergence_threshold}")
                logger.info(f"    - batch_size: {self.repetition_batch_size}")
                if self.repetition_multi_scale:
                    logger.info(f"  Multi-scale parameters:")
                    logger.info(f"    - enabled: True")
                    logger.info(f"    - scale_levels: {self.repetition_scale_levels}")
                if self.repetition_patch_aware:
                    logger.info(f"  Patch-aware parameters:")
                    logger.info(f"    - enabled: True")
                    logger.info(f"    - num_windows: {self.repetition_num_windows}")
                if self.repetition_boundary_aware:
                    logger.info(f"  Boundary-spanning parameters:")
                    logger.info(f"    - enabled: True")
                    logger.info(f"    - span_before: {self.repetition_boundary_span_before} bytes")
                    logger.info(f"    - span_after: {self.repetition_boundary_span_after} bytes")
                    logger.info(f"    - min_match: {self.repetition_boundary_min_match} bytes")
                
                # Recursive repetition detection and entropy adjustment with hierarchical layers
                tokens_for_detection = tokens[0] if tokens.dim() == 2 else tokens
                original_scores = scores.clone()
                previous_entropy_mean = scores.mean().item()
                
                # Store original patch boundaries before any entropy adjustment
                # These will be used for boundary-spanning detection to identify which boundaries should be merged
                original_patch_starts = None
                if self.repetition_boundary_aware:
                    original_patch_starts = find_entropy_patch_start_ids(
                        original_scores,
                        threshold=self.threshold,
                        threshold_add=self.threshold_add,
                        monotonicity=self.monotonicity,
                        max_patch_length=self.max_patch_length,
                    )
                    if original_patch_starts.shape[0] > 0 and original_patch_starts.shape[1] > 0:
                        original_patch_count = original_patch_starts.shape[1]
                        logger.info(f"  Original patch boundaries: {original_patch_count} boundaries before entropy adjustment")
                    else:
                        original_patch_starts = None
                
                for iteration in range(self.repetition_max_iterations + 1):
                    iteration_label = "Initial" if iteration == 0 else f"Iteration {iteration}"
                    logger.info("")
                    logger.info(f"  {iteration_label} pass:")
                    logger.info(f"    Current entropy stats: mean={scores.mean().item():.4f}, "
                              f"min={scores.min().item():.4f}, max={scores.max().item():.4f}, "
                              f"std={scores.std().item():.4f}")
                    
                    all_repetition_pairs = []
                    
                    # Layer A: Pattern Discovery
                    if self.repetition_multi_scale:
                        logger.info("    Layer A: Multi-scale pattern discovery")
                        multi_scale_pairs = detect_repetitions_multi_scale(
                            tokens_for_detection,
                            scale_levels=self.repetition_scale_levels,
                            max_distance=self.repetition_max_distance,
                            window_size=self.repetition_window_size,
                            max_repetitions=self.repetition_max_pairs,
                            sort_by_length=self.repetition_sort_by_length,
                        )
                        all_repetition_pairs.extend(multi_scale_pairs)
                        logger.info(f"      Found {len(multi_scale_pairs)} repetition pair(s) at multiple scales")
                    else:
                        # Single-scale detection
                        logger.info("    Layer A: Single-scale pattern discovery")
                        single_scale_pairs = detect_repetitions(
                            tokens_for_detection,
                            min_match_length=self.repetition_min_match,
                            max_distance=self.repetition_max_distance,
                            window_size=self.repetition_window_size,
                            hash_size=self.repetition_hash_size,
                            max_repetitions=self.repetition_max_pairs,
                            sort_by_length=self.repetition_sort_by_length,
                        )
                        all_repetition_pairs.extend(single_scale_pairs)
                        logger.info(f"      Found {len(single_scale_pairs)} repetition pair(s)")
                    
                    # Layer B: Patch-Aware Detection (if enabled and we have patches)
                    if self.repetition_patch_aware and iteration > 0:
                        logger.info("    Layer B: Patch-aware detection")
                        # Get current patch boundaries from scores (approximate)
                        # We'll need to do a quick patch to identify common patches
                        # For now, we'll skip this in the first iteration and do it in later iterations
                        try:
                            # Quick patch to identify common patches
                            patch_starts = find_entropy_patch_start_ids(
                                scores,
                                threshold=self.threshold,
                                threshold_add=self.threshold_add,
                                monotonicity=self.monotonicity,
                                max_patch_length=self.max_patch_length,
                            )
                            if patch_starts.shape[0] > 0 and patch_starts.shape[1] > 0:
                                # Calculate patch lengths from patch start positions
                                patch_starts_1d = patch_starts[0] if patch_starts.dim() > 1 else patch_starts
                                # Convert boolean mask to indices
                                if patch_starts_1d.dtype == torch.bool:
                                    patch_start_indices = torch.where(patch_starts_1d)[0]
                                else:
                                    patch_start_indices = patch_starts_1d
                                
                                if len(patch_start_indices) > 0:
                                    # Calculate lengths between consecutive patch starts
                                    patch_start_indices = torch.cat([
                                        patch_start_indices,
                                        torch.tensor([seq_len], device=patch_start_indices.device, dtype=patch_start_indices.dtype)
                                    ])
                                    patch_lengths = torch.diff(patch_start_indices)
                                    
                                    # Identify common patches
                                    if len(patch_lengths) > 0:
                                        common_patches = identify_common_patches(
                                            patch_lengths,
                                            tokens_for_detection,
                                            min_occurrences=3,
                                            min_patch_length=8,
                                        )
                                    else:
                                        common_patches = []
                                else:
                                    common_patches = []
                                
                                if common_patches:
                                    # Extract multi-windows around common patches
                                    patch_aware_pairs = extract_multi_windows_around_patches(
                                        tokens_for_detection,
                                        common_patches,
                                        num_windows=self.repetition_num_windows,
                                        window_size=32,
                                    )
                                    all_repetition_pairs.extend(patch_aware_pairs)
                                    logger.info(f"      Found {len(patch_aware_pairs)} window pair(s) around {len(common_patches)} common patch(es)")
                        except Exception as e:
                            logger.debug(f"      Patch-aware detection skipped: {e}")
                    
                    # Layer C: Boundary-Spanning Detection (if enabled and we have patches)
                    if self.repetition_boundary_aware and iteration > 0:
                        logger.info("    Layer C: Boundary-spanning repetition detection")
                        try:
                            # Use original patch boundaries (not adjusted) to identify which boundaries should be merged
                            # This allows us to detect unnecessary splits that occurred before entropy adjustment
                            if original_patch_starts is not None and original_patch_starts.shape[0] > 0 and original_patch_starts.shape[1] > 0:
                                # Detect repetitions of sequences spanning patch boundaries
                                boundary_spanning_pairs = detect_boundary_spanning_repetitions(
                                    tokens_for_detection,
                                    original_patch_starts,  # Use original boundaries, not adjusted
                                    span_before=self.repetition_boundary_span_before,
                                    span_after=self.repetition_boundary_span_after,
                                    min_match_length=self.repetition_boundary_min_match,
                                    max_distance=self.repetition_max_distance,
                                    max_repetitions=self.repetition_max_pairs,
                                    hash_size=self.repetition_hash_size,
                                )
                                all_repetition_pairs.extend(boundary_spanning_pairs)
                                logger.info(f"      Found {len(boundary_spanning_pairs)} boundary-spanning repetition pair(s)")
                                
                                # Directly reduce entropy differences at boundaries to enable merging
                                if boundary_spanning_pairs:
                                    scores = reduce_entropy_differences_at_boundaries(
                                        scores,
                                        tokens,
                                        boundary_spanning_pairs,
                                        original_patch_starts,
                                        threshold=self.threshold,
                                        reduction_factor=0.5,  # Reduce differences by 50%
                                    )
                            else:
                                logger.debug("      No original patch boundaries available for boundary-spanning detection")
                        except Exception as e:
                            logger.debug(f"      Boundary-spanning detection skipped: {e}")
                    
                    # Deduplicate all repetition pairs
                    if all_repetition_pairs:
                        from bytelatent.data.repetition_detector import _deduplicate_matches
                        repetition_pairs = _deduplicate_matches(all_repetition_pairs)
                    else:
                        repetition_pairs = []
                    
                    if not repetition_pairs:
                        logger.info(f"    No repetitions found in {iteration_label.lower()}")
                        if iteration == 0:
                            logger.info("    Using original entropies (no repetitions detected)")
                        else:
                            logger.info(f"    Converged after {iteration} iteration(s) - no new repetitions")
                        break
                    
                    logger.info(f"    Found {len(repetition_pairs)} repetition pair(s)")
                    
                    # Log details about first few repetitions
                    for idx, (s1, e1, s2, e2) in enumerate(repetition_pairs[:3], 1):
                        match_len = e1 - s1
                        distance = abs(s2 - s1)
                        avg_entropy_before = scores[0, s1:e1].mean().item() if scores.dim() == 2 else scores[s1:e1].mean().item()
                        logger.info(f"      Rep {idx}: length={match_len}B, distance={distance}B, "
                                   f"avg_entropy={avg_entropy_before:.4f}, "
                                   f"positions=[{s1}:{e1}] and [{s2}:{e2}]")
                    if len(repetition_pairs) > 3:
                        logger.info(f"      ... and {len(repetition_pairs) - 3} more repetition(s)")
                    
                    # Recalculate entropies for repetitive regions
                    scores_before = scores.clone()
                    scores = recalculate_entropies_for_repetitions(
                        scores,
                        tokens,
                        repetition_pairs,
                        self.entropy_model,
                        window_size=self.repetition_window_size,
                        device=self.device,
                        batch_size=self.repetition_batch_size,  # Configurable batch size
                        max_context_window=512,  # Use full 512-byte context window
                    )
                    
                    # Calculate entropy change
                    entropy_change = (scores_before.mean() - scores.mean()).item()
                    logger.info(f"    Entropy recalculation complete")
                    logger.info(f"    Entropy change: {entropy_change:+.6f} (mean decreased by {abs(entropy_change):.6f})")
                    
                    # Check convergence
                    if iteration > 0:
                        if abs(entropy_change) < self.repetition_convergence_threshold:
                            logger.info(f"    Converged after {iteration} iteration(s) - entropy change below threshold")
                            break
                    
                    # Check if we've reached max iterations
                    if iteration >= self.repetition_max_iterations:
                        logger.info(f"    Reached maximum iterations ({self.repetition_max_iterations})")
                        break
                
                # Final statistics
                total_entropy_reduction = (original_scores.mean() - scores.mean()).item()
                logger.info("")
                logger.info(f"  Final results:")
                logger.info(f"    Total entropy reduction: {total_entropy_reduction:.6f}")
                logger.info(f"    Final entropy stats: mean={scores.mean().item():.4f}, "
                          f"min={scores.min().item():.4f}, max={scores.max().item():.4f}")
                
                # Compare original vs final boundary counts if boundary-aware detection was used
                if self.repetition_boundary_aware and original_patch_starts is not None:
                    final_patch_starts = find_entropy_patch_start_ids(
                        scores,
                        threshold=self.threshold,
                        threshold_add=self.threshold_add,
                        monotonicity=self.monotonicity,
                        max_patch_length=self.max_patch_length,
                    )
                    if final_patch_starts.shape[0] > 0 and final_patch_starts.shape[1] > 0:
                        original_count = original_patch_starts.shape[1]
                        final_count = final_patch_starts.shape[1]
                        boundaries_merged = original_count - final_count
                        logger.info(f"    Boundary comparison:")
                        logger.info(f"      Original boundaries: {original_count}")
                        logger.info(f"      Final boundaries: {final_count}")
                        if original_count > 0:
                            logger.info(f"      Boundaries merged: {boundaries_merged} ({boundaries_merged/original_count*100:.1f}% reduction)")
                            
                            # Log entropy difference statistics at boundaries
                            if original_patch_starts.shape[0] > 0:
                                original_starts_1d = original_patch_starts[0] if original_patch_starts.dim() > 1 else original_patch_starts
                                if original_starts_1d.dtype == torch.bool:
                                    original_indices = torch.where(original_starts_1d)[0].cpu().numpy()
                                else:
                                    original_indices = original_starts_1d.cpu().numpy()
                                
                                # Calculate entropy differences at original boundaries
                                scores_1d = scores[0] if scores.dim() == 2 else scores
                                boundary_diffs = []
                                for i in range(1, len(original_indices)):
                                    if original_indices[i] < scores_1d.shape[0] and original_indices[i] > 0:
                                        diff = (scores_1d[original_indices[i]] - scores_1d[original_indices[i] - 1]).item()
                                        boundary_diffs.append(diff)
                                
                                if boundary_diffs:
                                    import numpy as np
                                    logger.info(f"      Entropy differences at boundaries:")
                                    logger.info(f"        Mean: {np.mean(boundary_diffs):.4f}")
                                    logger.info(f"        Min: {np.min(boundary_diffs):.4f}")
                                    logger.info(f"        Max: {np.max(boundary_diffs):.4f}")
                                    logger.info(f"        Threshold: {self.threshold:.4f}")
                                    above_threshold = sum(1 for d in boundary_diffs if d > self.threshold)
                                    logger.info(f"        Above threshold: {above_threshold}/{len(boundary_diffs)} ({above_threshold/len(boundary_diffs)*100:.1f}%)")
                        else:
                            logger.info(f"      Boundaries merged: 0")
                
                logger.info("=" * 80)
                
                if self.log_time:
                    self.log["repetition_detection"] += time.time() - rep_start
                    s = time.time()
            
            patch_start_ids = find_entropy_patch_start_ids(
                scores,
                self.patch_size,
                include_next_token=include_next_token,
                threshold=threshold if threshold is not None else self.threshold,
                threshold_add=self.threshold_add,
                monotonicity=self.monotonicity,
                max_patch_length=self.max_patch_length,
            )
            if self.log_time:
                self.log["find_entropy_patch_start_ids"] += time.time() - s
                s = time.time()
            patch_lengths = patch_lengths_from_start_ids(
                patch_start_ids, seq_len_next_tok
            )
            if self.log_time:
                self.log["patch_lengths_from_start_ids"] += time.time() - s
                s = time.time()
        # BPE
        elif self.patching_mode == PatchingModeEnum.bpe:
            patch_start_ids = find_bpe_delim_patch_start_ids(tokens, delim=BPE_ID)
            patch_lengths = patch_lengths_from_start_ids(
                patch_start_ids, seq_len_next_tok
            )
        elif self.patching_mode == PatchingModeEnum.bpe_patcher:
            patch_start_ids = find_bpe_patcher_patch_start_ids(
                tokens,
                self.entropy_model,
                self.patching_batch_size,
                self.device,
                include_next_token,
            )
            patch_lengths = patch_lengths_from_start_ids(
                patch_start_ids, seq_len_next_tok
            )
        # SPACE
        elif self.patching_mode == PatchingModeEnum.space:
            patch_start_ids = find_space_patch_start_ids(tokens)
            patch_lengths = patch_lengths_from_start_ids(
                patch_start_ids, seq_len_next_tok
            )
        else:
            raise NotImplementedError(f"self.patching_mode {self.patching_mode}")

        # Apply any processing to patch lengths
        if self.max_patch_length is not None:
            # TODO: avoid going back to a list here.
            patch_lengths = [
                split_large_numbers(pl, self.max_patch_length)
                for pl in patch_lengths.tolist()
            ]
            max_len = max([len(pl) for pl in patch_lengths])
            patch_lengths = [rightpad(pl, 0, max_len=max_len) for pl in patch_lengths]
            patch_lengths = torch.tensor(
                patch_lengths, dtype=tokens.dtype, device=tokens.device
            )
        assert not check_non_zero_after_zero(patch_lengths)
        # Find the last non-zero column index using argmax on a reversed version of the tensor
        last_non_zero_col_reversed = (
            (patch_lengths != 0).flip(dims=[1]).int().argmax(dim=1).min()
        )
        # Slice the tensor up to the last non-zero column
        patch_lengths = patch_lengths[
            :, : patch_lengths.shape[1] - last_non_zero_col_reversed
        ]
        assert (
            torch.sum(patch_lengths)
            == tokens.numel() + include_next_token * tokens.shape[0]
        ), f"{torch.sum(patch_lengths)} != {tokens.numel() + include_next_token * tokens.shape[0]}"
        if self.log_time:
            self.log["postprocessing_patch_lengths"] += time.time() - s
            self.log["tokens"] += patch_lengths.sum().item()
        return patch_lengths, scores
