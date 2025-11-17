"""
Strict monotonicity patching: Main patching pipeline with automatic chunking.

The main function is `strict_monotonicity_patch()` which uses entropy-based patching
with strict monotonicity constraint (threshold ≥1.35). It automatically handles chunking
for large sequences (>8192 bytes) and is optimized for production use.

Deprecated functions (cross_reference_patch, aggressive_ngram_merge, three_stage_patch)
are kept for reference but not used in the main pipeline.
"""

import logging
import time
from typing import List, Tuple
import torch
import numpy as np

from bytelatent.data.patcher import Patcher, PatcherArgs, calculate_entropies, find_entropy_patch_start_ids
from bytelatent.data.repetition_detector import detect_repetitions

logger = logging.getLogger(__name__)


# Deprecated: Stage 2/3 functions kept for reference only
def simulate_patch_length_in_window(
    tokens: torch.Tensor,
    entropy_model,
    sequence_start: int,
    sequence_end: int,
    all_occurrence_starts: List[int],
    all_occurrence_ends: List[int],
    threshold: float,
    device: str,
) -> int:
    """
    Simulate what patch length would be if all occurrences of a sequence were in the same 512-byte window.
    
    Args:
        tokens: Full token tensor
        entropy_model: Entropy model
        sequence_start: Start of the sequence pattern
        sequence_end: End of the sequence pattern
        all_occurrence_starts: List of start positions for all occurrences
        all_occurrence_ends: List of end positions for all occurrences
        threshold: Monotonicity threshold
        device: Device to run on
    
    Returns:
        Simulated patch length if all occurrences were in same window
    """
    # Extract the sequence pattern
    seq_len = sequence_end - sequence_start
    if seq_len == 0:
        return 0
    
    # Create a simulated 512-byte window with multiple occurrences
    # We'll pack as many occurrences as possible into 512 bytes
    max_window = 512
    simulated_tokens = []
    
    # Add occurrences until we hit 512 bytes
    for start, end in zip(all_occurrence_starts, all_occurrence_ends):
        if len(simulated_tokens) + seq_len > max_window:
            break
        occurrence = tokens[start:end]
        simulated_tokens.append(occurrence)
    
    if not simulated_tokens:
        return seq_len
    
    # Concatenate all occurrences
    simulated_window = torch.cat(simulated_tokens)
    
    # Pad or truncate to exactly 512 bytes
    if simulated_window.shape[0] < max_window:
        pad_size = max_window - simulated_window.shape[0]
        pad = torch.zeros(pad_size, dtype=simulated_window.dtype, device=simulated_window.device)
        simulated_window = torch.cat([simulated_window, pad])
    elif simulated_window.shape[0] > max_window:
        simulated_window = simulated_window[:max_window]
    
    # Calculate entropies for simulated window
    simulated_window = simulated_window.unsqueeze(0)  # Add batch dimension [1, 512]
    simulated_window = simulated_window.to(device)
    
    with torch.no_grad():
        # Entropy model expects [batch, seq_len] format
        pred = entropy_model(simulated_window)
        # pred shape: [1, 512, vocab_size]
        from bytelatent.data.patcher import entropy
        entropies = entropy(pred)  # [1, 512]
    
    # Calculate patch boundaries with monotonicity
    patch_starts = find_entropy_patch_start_ids(
        entropies,
        threshold=threshold,
        monotonicity=True,
        max_patch_length=None,
    )
    
    # Find how long the first patch would be (which should contain the repeated sequence)
    # With multiple occurrences in the same window, monotonicity should create one long patch
    if patch_starts.shape[0] > 0 and patch_starts.shape[1] > 1:
        # First patch goes from start to second patch start
        first_patch_end = patch_starts[0, 1].item() if patch_starts.shape[1] > 1 else max_window
        simulated_patch_len = min(first_patch_end, max_window)
        # If we have multiple occurrences packed, the patch should be at least the sum of occurrences
        # (but capped by window size)
        min_expected = min(len(simulated_tokens) * seq_len, max_window)
        return max(simulated_patch_len, min_expected)
    
    # Fallback: if no boundaries found, the whole window is one patch
    return min(max_window, len(simulated_tokens) * seq_len)


# Deprecated: Stage 2 function kept for reference only
def cross_reference_patch(
    tokens: torch.Tensor,
    entropy_model,
    initial_patch_lengths: torch.Tensor,
    initial_patch_starts: List[int],
    threshold: float = 1.35,
    min_repetitions: int = 3,
    min_match_length: int = 8,
    device: str = "cpu",
) -> Tuple[torch.Tensor, List[int]]:
    """
    Two-stage cross-reference patching:
    1. Use initial patches from strict monotonicity patching
    2. Find repetitive sequences that would be longer patches if in same 512-byte window
    3. Merge patches accordingly
    
    Args:
        tokens: Token tensor [batch_size, seq_len] or [seq_len]
        entropy_model: Entropy model
        initial_patch_lengths: Initial patch lengths from stage 1
        initial_patch_starts: Initial patch start positions
        threshold: Monotonicity threshold (default: 1.35 for strict)
        min_repetitions: Minimum number of repetitions to consider (default: 3)
        min_match_length: Minimum match length (default: 8)
        device: Device to run on
    
    Returns:
        Updated patch lengths and patch starts
    """
    # Work with 1D tensor
    if tokens.dim() == 2:
        tokens_1d = tokens[0]
    else:
        tokens_1d = tokens
    
    seq_len = tokens_1d.shape[0]
    logger.info(f"Cross-reference patching: {seq_len} bytes, {len(initial_patch_starts)} initial patches")
    
    # Stage 2: Find repetitive sequences across initial patches
    logger.info("Stage 2: Finding repetitive sequences that would benefit from merging")
    
    # Detect repetitions in the full sequence
    repetition_pairs = detect_repetitions(
        tokens_1d,
        min_match_length=min_match_length,
        max_distance=None,  # Unlimited distance
        max_repetitions=None,  # Get all
        sort_by_length=True,
    )
    
    # Group repetitions by sequence content
    from collections import defaultdict
    sequence_groups = defaultdict(list)
    
    for start1, end1, start2, end2 in repetition_pairs:
        # Extract the sequence bytes
        seq_bytes = bytes(tokens_1d[start1:end1].cpu().numpy())
        sequence_groups[seq_bytes].append((start1, end1))
        sequence_groups[seq_bytes].append((start2, end2))
    
    # Deduplicate occurrence positions
    for seq_bytes in sequence_groups:
        occurrences = sorted(set(sequence_groups[seq_bytes]))
        sequence_groups[seq_bytes] = occurrences
    
    # Filter to sequences with enough repetitions
    candidate_sequences = {
        seq_bytes: occurrences
        for seq_bytes, occurrences in sequence_groups.items()
        if len(occurrences) >= min_repetitions
    }
    
    logger.info(f"Found {len(candidate_sequences)} candidate sequence(s) with {min_repetitions}+ repetitions")
    
    # Build mapping from position to patch index
    position_to_patch = {}
    for patch_idx, patch_start in enumerate(initial_patch_starts):
        patch_length = initial_patch_lengths[patch_idx].item()
        patch_end = patch_start + patch_length
        for pos in range(patch_start, patch_end):
            position_to_patch[pos] = patch_idx
    
    # For each candidate sequence, simulate patch length in same window and mark for merging
    patches_to_merge = {}  # patch_idx -> set of patches to merge with
    sequences_to_merge = []  # List of (seq_bytes, occurrences, simulated_length) for logging
    
    for seq_bytes, occurrences in candidate_sequences.items():
        if len(occurrences) < 2:
            continue
        
        # Get first occurrence as reference
        first_start, first_end = occurrences[0]
        seq_len_actual = first_end - first_start
        
        # Get all occurrence positions
        all_starts = [start for start, _ in occurrences]
        all_ends = [end for _, end in occurrences]
        
        # Simulate what patch length would be if all in same window
        simulated_length = simulate_patch_length_in_window(
            tokens_1d,
            entropy_model,
            first_start,
            first_end,
            all_starts,
            all_ends,
            threshold,
            device,
        )
        
        # If simulated length is significantly longer than individual occurrence, merge
        # Use a more lenient threshold - if simulated length is longer, it means repetition helps
        merge_threshold = max(seq_len_actual * 1.2, seq_len_actual + 8)  # At least 20% longer or 8 bytes more
        if simulated_length > merge_threshold:
            logger.info(f"  Sequence {seq_len_actual}B appears {len(occurrences)} times, "
                       f"would be {simulated_length}B patch in same window - merging")
            sequences_to_merge.append((seq_bytes, occurrences, simulated_length))
            
            # Find all patches that contain these occurrences
            # Group occurrences by which patches they're in
            occurrence_patches = []
            for start, end in occurrences:
                # Find the patch that contains the start of this occurrence
                patch_idx = position_to_patch.get(start, None)
                if patch_idx is not None:
                    occurrence_patches.append((start, end, patch_idx))
            
            # For each occurrence, we want to merge the patch it's in
            # If multiple occurrences are in the same patch, that's even better
            patches_with_occurrences = {}
            for start, end, patch_idx in occurrence_patches:
                if patch_idx not in patches_with_occurrences:
                    patches_with_occurrences[patch_idx] = []
                patches_with_occurrences[patch_idx].append((start, end))
            
            # Merge patches that contain occurrences of this repetitive sequence
            # Strategy: For each patch containing an occurrence, extend it to include
            # adjacent patches if they're close enough (within the simulated length)
            if len(patches_with_occurrences) > 1:
                # Sort patches by position
                sorted_patches = sorted(patches_with_occurrences.keys())
                
                # Group consecutive patches that should be merged
                # We'll merge patches that are within the simulated length of each other
                merge_groups = []
                current_group = [sorted_patches[0]]
                
                for i in range(1, len(sorted_patches)):
                    prev_patch = sorted_patches[i - 1]
                    curr_patch = sorted_patches[i]
                    
                    # Check if patches are close enough to merge
                    prev_end = initial_patch_starts[prev_patch] + initial_patch_lengths[prev_patch].item()
                    curr_start = initial_patch_starts[curr_patch]
                    gap = curr_start - prev_end
                    
                    # If gap is small relative to simulated length, merge
                    if gap < simulated_length * 0.5:  # Within 50% of simulated length
                        current_group.append(curr_patch)
                    else:
                        merge_groups.append(current_group)
                        current_group = [curr_patch]
                
                if current_group:
                    merge_groups.append(current_group)
                
                # For each merge group, use the first patch as target
                for merge_group in merge_groups:
                    if len(merge_group) > 1:
                        merge_target = merge_group[0]
                        for patch_idx in merge_group[1:]:
                            if merge_target not in patches_to_merge:
                                patches_to_merge[merge_target] = set()
                            patches_to_merge[merge_target].add(patch_idx)
    
    logger.info(f"Found {len(sequences_to_merge)} sequence(s) to merge, affecting {sum(len(v) for v in patches_to_merge.values())} patch(es)")
    
    # Consolidate merge groups (handle transitive merges)
    # If patch A merges with B, and B merges with C, then A should merge with both B and C
    # Use union-find style approach to group all patches that should be merged together
    merge_groups = {}
    patch_to_group = {}
    
    def find_group(patch_idx):
        """Find the root group for a patch."""
        if patch_idx not in patch_to_group:
            patch_to_group[patch_idx] = patch_idx
            merge_groups[patch_idx] = {patch_idx}
        if patch_to_group[patch_idx] != patch_idx:
            # Path compression
            root = find_group(patch_to_group[patch_idx])
            patch_to_group[patch_idx] = root
        return patch_to_group[patch_idx]
    
    def union_groups(patch1, patch2):
        """Merge two patch groups."""
        root1 = find_group(patch1)
        root2 = find_group(patch2)
        if root1 != root2:
            # Merge group2 into group1
            merge_groups[root1].update(merge_groups[root2])
            patch_to_group[root2] = root1
            del merge_groups[root2]
    
    # Build merge groups
    for merge_target, patches_to_merge_with in patches_to_merge.items():
        for patch in patches_to_merge_with:
            union_groups(merge_target, patch)
    
    # Convert to final merge structure (use minimum patch index as target)
    consolidated_merges = {}
    for root, group in merge_groups.items():
        if len(group) > 1:
            # Use minimum patch index as merge target
            merge_target = min(group)
            merged_patches = group - {merge_target}
            if merged_patches:
                consolidated_merges[merge_target] = merged_patches
    
    # Use consolidated merges
    patches_to_merge = consolidated_merges
    
    # Build a set of all patches that will be removed (merged into others)
    patches_to_remove = set()
    for merge_target, patches_to_merge_with in patches_to_merge.items():
        patches_to_remove.update(patches_to_merge_with)
    
    # Create new patch structure
    final_patch_starts = []
    final_patch_lengths = []
    
    i = 0
    while i < len(initial_patch_starts):
        if i in patches_to_remove:
            # This patch will be merged, skip it
            i += 1
            continue
        
        # Start of a new (possibly merged) patch
        current_start = initial_patch_starts[i]
        current_length = initial_patch_lengths[i].item()
        
        # Check if this patch is a merge target
        if i in patches_to_merge:
            # Merge with all patches in the merge set
            patches_to_merge_with = sorted(patches_to_merge[i])
            
            # Find the end of the last patch to merge
            merge_end_idx = max([i] + list(patches_to_merge_with))
            
            # Merge all consecutive patches from i to merge_end_idx (inclusive)
            # This ensures we merge everything in between, not just the explicitly marked ones
            merge_end_pos = initial_patch_starts[merge_end_idx] + initial_patch_lengths[merge_end_idx].item()
            
            # Calculate merged length
            merged_length = merge_end_pos - current_start
            
            final_patch_starts.append(current_start)
            final_patch_lengths.append(merged_length)
            
            # Skip all merged patches (from i+1 to merge_end_idx, inclusive)
            i = merge_end_idx + 1
        else:
            # Regular patch, keep as is
            final_patch_starts.append(current_start)
            final_patch_lengths.append(current_length)
            i += 1
    
    # Convert to tensors
    final_patch_lengths_tensor = torch.tensor(
        final_patch_lengths,
        dtype=initial_patch_lengths.dtype,
        device=initial_patch_lengths.device,
    )
    
    logger.info(f"Merged {len(initial_patch_starts)} patches into {len(final_patch_starts)} patches")
    logger.info(f"  Patches removed: {len(patches_to_remove)}")
    logger.info(f"  Average patch length: {final_patch_lengths_tensor.float().mean().item():.2f} bytes (was {initial_patch_lengths.float().mean().item():.2f})")
    
    return final_patch_lengths_tensor, final_patch_starts


# Deprecated: Stage 3 function kept for reference only
def aggressive_ngram_merge(
    tokens: torch.Tensor,
    entropy_model,
    current_patch_lengths: torch.Tensor,
    current_patch_starts: List[int],
    min_ngram_size: int = 5,
    min_repetitions: int = 2,
    scale_factor: float = 1.5,
    max_iterations: int = 5,
    convergence_threshold: int = 0,
    max_repetitions: int | None = None,
    max_candidate_sequences: int = 1000,
    device: str = "cpu",
) -> Tuple[torch.Tensor, List[int]]:
    """
    Aggressive n-gram based recursive merging (Stage 3).
    
    Operates like n-gram analysis: longer byte sequences (5+ bytes) that repeat more
    frequently should be allowed to form proportionally longer patches.
    
    Key formula: allowed_patch_length = sequence_length * repetition_count * scale_factor
    
    Args:
        tokens: Token tensor [batch_size, seq_len] or [seq_len]
        entropy_model: Entropy model (not used in this stage, but kept for consistency)
        current_patch_lengths: Current patch lengths from previous stages
        current_patch_starts: Current patch start positions
        min_ngram_size: Minimum sequence length to consider (default: 5, like n-gram 5+)
        min_repetitions: Minimum repetition count (default: 2, more aggressive than Stage 2's 3)
        scale_factor: Multiplier for allowed patch length (default: 1.5)
        max_iterations: Maximum recursive passes (default: 5, reduced for large sequences)
        convergence_threshold: Stop if merge count < this (default: 0)
        max_repetitions: Maximum repetition pairs to process (None = all, auto-limited for large sequences)
        max_candidate_sequences: Maximum candidate sequences to process (default: 1000, auto-limited for large sequences)
        device: Device to run on
    
    Returns:
        Updated patch lengths and patch starts after aggressive merging
    """
    # Work with 1D tensor
    if tokens.dim() == 2:
        tokens_1d = tokens[0]
    else:
        tokens_1d = tokens
    
    seq_len = tokens_1d.shape[0]
    logger.info(f"Aggressive n-gram merging: {seq_len} bytes, {len(current_patch_starts)} current patches")
    logger.info(f"  Parameters: min_ngram_size={min_ngram_size}, min_repetitions={min_repetitions}, "
               f"scale_factor={scale_factor}, max_iterations={max_iterations}")
    
    # For very large sequences, limit processing to avoid excessive computation
    if seq_len > 1000000:  # > 1MB
        logger.info(f"  Very large sequence detected ({seq_len} bytes) - applying aggressive optimizations")
        # For very large sequences, consider skipping or heavily limiting
        if max_repetitions is None:
            max_repetitions = 500  # Very aggressive limit (was 1000)
        if max_candidate_sequences > 50:
            max_candidate_sequences = 50  # Very limited candidates (was 100)
        if max_iterations > 1:
            max_iterations = 1  # Only 1 iteration (was 2)
        if min_ngram_size < 20:
            min_ngram_size = 20  # Increase min size significantly (was 16)
            logger.info(f"    Increased min_ngram_size to {min_ngram_size} for performance")
        # Also increase min_repetitions to be more selective
        if min_repetitions < 3:
            min_repetitions = 3
            logger.info(f"    Increased min_repetitions to {min_repetitions} for performance")
    elif seq_len > 500000:  # > 500KB
        logger.info(f"  Large sequence detected ({seq_len} bytes) - applying performance optimizations")
        if max_repetitions is None:
            max_repetitions = 2000  # More aggressive limit
        if max_candidate_sequences > 200:
            max_candidate_sequences = 200  # More limited candidates
        if max_iterations > 2:
            max_iterations = 2  # Reduce iterations
        if min_ngram_size < 12:
            min_ngram_size = 12  # Increase min size
            logger.info(f"    Increased min_ngram_size to {min_ngram_size} for performance")
    elif seq_len > 100000:  # > 100KB
        logger.info(f"  Medium sequence detected ({seq_len} bytes) - applying moderate optimizations")
        if max_repetitions is None:
            max_repetitions = 5000
        if max_candidate_sequences > 500:
            max_candidate_sequences = 500
        if max_iterations > 3:
            max_iterations = 3
    
    # Convert to numpy for easier manipulation
    tokens_np = tokens_1d.cpu().numpy()
    
    # Recursive merging loop
    patch_lengths = current_patch_lengths.clone()
    patch_starts = current_patch_starts.copy()
    
    total_merged = 0
    
    # Cache repetition detection results - only re-detect if patches changed significantly
    cached_repetition_pairs = None
    cached_patch_count = len(patch_starts)
    
    for iteration in range(max_iterations):
        iter_start_time = time.time()
        logger.info(f"  Iteration {iteration + 1}/{max_iterations}")
        
        # Build mapping from position to patch index (optimized for large sequences)
        step_start = time.time()
        # For large sequences, use binary search instead of full mapping
        if seq_len > 500000:
            # Use binary search function instead of building full dict
            def find_patch_for_position(pos):
                # Binary search for patch containing position
                left, right = 0, len(patch_starts) - 1
                while left <= right:
                    mid = (left + right) // 2
                    patch_start = patch_starts[mid]
                    patch_end = patch_start + patch_lengths[mid].item()
                    if patch_start <= pos < patch_end:
                        return mid
                    elif pos < patch_start:
                        right = mid - 1
                    else:
                        left = mid + 1
                return None
            position_to_patch_func = find_patch_for_position
        else:
            # For smaller sequences, build full mapping
            position_to_patch = {}
            for patch_idx, patch_start in enumerate(patch_starts):
                patch_length = patch_lengths[patch_idx].item()
                patch_end = patch_start + patch_length
                for pos in range(patch_start, patch_end):
                    position_to_patch[pos] = patch_idx
            position_to_patch_func = lambda pos: position_to_patch.get(pos)
        
        step_time = time.time() - step_start
        logger.info(f"    Step 1 (position mapping): {step_time:.2f}s")
        
        # Detect all sequences of length >= min_ngram_size that appear multiple times
        # For large sequences, cache results and only re-detect if patches changed significantly
        step_start = time.time()
        
        # For large sequences, skip re-detection after first iteration if patches haven't changed much
        current_patch_count = len(patch_starts)
        patch_change_ratio = abs(current_patch_count - cached_patch_count) / max(cached_patch_count, 1)
        
        if cached_repetition_pairs is not None and seq_len > 500000:
            # For large sequences, reuse cached results if patches changed < 5%
            if patch_change_ratio < 0.05:
                logger.info(f"    Reusing cached repetition detection (patch change: {patch_change_ratio*100:.1f}%)")
                repetition_pairs = cached_repetition_pairs
            else:
                logger.info(f"    Re-detecting repetitions (patch change: {patch_change_ratio*100:.1f}%)")
                repetition_pairs = detect_repetitions(
                    tokens_1d,
                    min_match_length=min_ngram_size,
                    max_distance=None,
                    max_repetitions=max_repetitions,
                    sort_by_length=True,
                )
                cached_repetition_pairs = repetition_pairs
                cached_patch_count = current_patch_count
        else:
            # First iteration or small sequence - always detect
            repetition_pairs = detect_repetitions(
                tokens_1d,
                min_match_length=min_ngram_size,
                max_distance=None,
                max_repetitions=max_repetitions,
                sort_by_length=True,
            )
            cached_repetition_pairs = repetition_pairs
            cached_patch_count = current_patch_count
        
        step_time = time.time() - step_start
        logger.info(f"    Step 2 (repetition detection): {step_time:.2f}s, found {len(repetition_pairs)} pairs")
        
        # Group repetitions by sequence content and count occurrences
        step_start = time.time()
        from collections import defaultdict
        sequence_groups = defaultdict(list)
        
        for start1, end1, start2, end2 in repetition_pairs:
            # Extract the sequence bytes
            seq_bytes = bytes(tokens_np[start1:end1])
            sequence_groups[seq_bytes].append((start1, end1))
            sequence_groups[seq_bytes].append((start2, end2))
        
        # Deduplicate occurrence positions
        for seq_bytes in sequence_groups:
            occurrences = sorted(set(sequence_groups[seq_bytes]))
            sequence_groups[seq_bytes] = occurrences
        step_time = time.time() - step_start
        logger.info(f"    Step 3 (grouping/deduplication): {step_time:.2f}s, {len(sequence_groups)} unique sequences")
        
        # Filter to sequences with enough repetitions
        step_start = time.time()
        candidate_sequences = {
            seq_bytes: occurrences
            for seq_bytes, occurrences in sequence_groups.items()
            if len(occurrences) >= min_repetitions
        }
        
        # Limit number of candidate sequences to process (prioritize by repetition count and length)
        if len(candidate_sequences) > max_candidate_sequences:
            # Sort by (repetition_count * sequence_length) descending, take top N
            sorted_candidates = sorted(
                candidate_sequences.items(),
                key=lambda x: (len(x[1]), len(x[0])),
                reverse=True
            )
            candidate_sequences = dict(sorted_candidates[:max_candidate_sequences])
            logger.info(f"    Limited to top {max_candidate_sequences} candidate sequences (by repetition count and length)")
        step_time = time.time() - step_start
        logger.info(f"    Step 4 (filtering/prioritization): {step_time:.2f}s")
        
        logger.info(f"    Found {len(candidate_sequences)} candidate sequence(s) with {min_repetitions}+ repetitions")
        
        if not candidate_sequences:
            logger.info(f"    No candidate sequences found - stopping early")
            break
        
        # For each candidate sequence, calculate allowed patch length and merge
        step_start = time.time()
        patches_to_merge = {}  # patch_idx -> set of patches to merge with
        sequences_merged = []
        
        for seq_bytes, occurrences in candidate_sequences.items():
            if len(occurrences) < min_repetitions:
                continue
            
            seq_len_actual = len(seq_bytes)
            repetition_count = len(occurrences)
            
            # Key formula: allowed_patch_length = sequence_length * repetition_count * scale_factor
            allowed_patch_length = int(seq_len_actual * repetition_count * scale_factor)
            
            # Find all patches containing these occurrences
            patches_containing_sequence = set()
            for start, end in occurrences:
                # Find patches that overlap with this occurrence
                # Sample positions to avoid checking every byte for large sequences
                if end - start > 100:
                    # For long sequences, sample positions
                    sample_positions = [start, (start + end) // 2, end - 1]
                else:
                    sample_positions = range(start, end)
                
                for pos in sample_positions:
                    patch_idx = position_to_patch_func(pos)
                    if patch_idx is not None:
                        patches_containing_sequence.add(patch_idx)
            
            # If we have multiple patches containing this sequence, consider merging
            if len(patches_containing_sequence) > 1:
                # Calculate total span of patches containing this sequence
                sorted_patch_indices = sorted(patches_containing_sequence)
                first_patch_start = patch_starts[sorted_patch_indices[0]]
                last_patch_idx = sorted_patch_indices[-1]
                last_patch_end = patch_starts[last_patch_idx] + patch_lengths[last_patch_idx].item()
                total_span = last_patch_end - first_patch_start
                
                # If total span is within allowed length, merge all patches
                if total_span <= allowed_patch_length:
                    merge_target = sorted_patch_indices[0]
                    for patch_idx in sorted_patch_indices[1:]:
                        if merge_target not in patches_to_merge:
                            patches_to_merge[merge_target] = set()
                        patches_to_merge[merge_target].add(patch_idx)
                    
                    sequences_merged.append((seq_len_actual, repetition_count, allowed_patch_length, total_span))
        step_time = time.time() - step_start
        logger.info(f"    Step 5 (sequence analysis): {step_time:.2f}s, analyzed {len(candidate_sequences)} sequences")
        
        if sequences_merged:
            logger.info(f"    Found {len(sequences_merged)} sequence(s) eligible for merging")
            # Log top 5 by allowed length
            top_sequences = sorted(sequences_merged, key=lambda x: x[2], reverse=True)[:5]
            for seq_len_act, rep_count, allowed_len, span in top_sequences:
                logger.info(f"      {seq_len_act}B sequence (×{rep_count}) → allowed {allowed_len}B, span {span}B")
        else:
            logger.info(f"    No sequences eligible for merging (spans exceed allowed length)")
        
        # Consolidate merge groups (handle transitive merges)
        step_start = time.time()
        merge_groups = {}
        patch_to_group = {}
        
        def find_group(patch_idx):
            """Find the root group for a patch."""
            if patch_idx not in patch_to_group:
                patch_to_group[patch_idx] = patch_idx
                merge_groups[patch_idx] = {patch_idx}
            if patch_to_group[patch_idx] != patch_idx:
                root = find_group(patch_to_group[patch_idx])
                patch_to_group[patch_idx] = root
            return patch_to_group[patch_idx]
        
        def union_groups(patch1, patch2):
            """Merge two patch groups."""
            root1 = find_group(patch1)
            root2 = find_group(patch2)
            if root1 != root2:
                merge_groups[root1].update(merge_groups[root2])
                patch_to_group[root2] = root1
                del merge_groups[root2]
        
        # Build merge groups
        for merge_target, patches_to_merge_with in patches_to_merge.items():
            for patch in patches_to_merge_with:
                union_groups(merge_target, patch)
        
        # Convert to final merge structure
        consolidated_merges = {}
        for root, group in merge_groups.items():
            if len(group) > 1:
                merge_target = min(group)
                merged_patches = group - {merge_target}
                if merged_patches:
                    consolidated_merges[merge_target] = merged_patches
        
        # Build set of patches to remove
        patches_to_remove = set()
        for merge_target, patches_to_merge_with in consolidated_merges.items():
            patches_to_remove.update(patches_to_merge_with)
        step_time = time.time() - step_start
        logger.info(f"    Step 6 (merge consolidation): {step_time:.2f}s, {len(consolidated_merges)} merge groups")
        
        if not patches_to_remove:
            logger.info(f"    No patches to merge in this iteration - convergence reached")
            break
        
        # Actually merge the patches
        step_start = time.time()
        new_patch_starts = []
        new_patch_lengths = []
        
        i = 0
        while i < len(patch_starts):
            if i in patches_to_remove:
                i += 1
                continue
            
            current_start = patch_starts[i]
            current_length = patch_lengths[i].item()
            
            if i in consolidated_merges:
                patches_to_merge_with = sorted(consolidated_merges[i])
                merge_end_idx = max([i] + list(patches_to_merge_with))
                merge_end_pos = patch_starts[merge_end_idx] + patch_lengths[merge_end_idx].item()
                merged_length = merge_end_pos - current_start
                
                new_patch_starts.append(current_start)
                new_patch_lengths.append(merged_length)
                i = merge_end_idx + 1
            else:
                new_patch_starts.append(current_start)
                new_patch_lengths.append(current_length)
                i += 1
        step_time = time.time() - step_start
        logger.info(f"    Step 7 (patch merging): {step_time:.2f}s")
        
        # Update for next iteration
        step_start = time.time()
        iteration_merged = len(patches_to_remove)
        total_merged += iteration_merged
        patch_starts = new_patch_starts
        patch_lengths = torch.tensor(
            new_patch_lengths,
            dtype=patch_lengths.dtype,
            device=patch_lengths.device,
        )
        # Update cached patch count for next iteration
        cached_patch_count = len(patch_starts)
        step_time = time.time() - step_start
        logger.info(f"    Step 8 (tensor conversion): {step_time:.2f}s")
        
        iter_time = time.time() - iter_start_time
        logger.info(f"    Merged {iteration_merged} patch(es) in this iteration")
        logger.info(f"    Iteration {iteration + 1} total time: {iter_time:.2f}s")
        logger.info(f"    New patch count: {len(patch_starts)} (was {len(patch_starts) + iteration_merged})")
        logger.info(f"    Average patch length: {patch_lengths.float().mean().item():.2f} bytes")
        
        # Check convergence
        if iteration_merged <= convergence_threshold:
            logger.info(f"    Convergence threshold reached ({iteration_merged} <= {convergence_threshold})")
            break
    
    logger.info(f"  Total patches merged across all iterations: {total_merged}")
    logger.info(f"  Final patch count: {len(patch_starts)} (started with {len(current_patch_starts)})")
    logger.info(f"  Final average patch length: {patch_lengths.float().mean().item():.2f} bytes "
               f"(started with {current_patch_lengths.float().mean().item():.2f})")
    
    return patch_lengths, patch_starts


def strict_monotonicity_patch(
    tokens: torch.Tensor,
    entropy_model,
    patcher_args: PatcherArgs,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Strict monotonicity patching pipeline.
    
    Uses entropy-based patching with strict monotonicity constraint (threshold ≥1.35).
    Optimized for large sequences with chunking support.
    
    Args:
        tokens: Token tensor [batch_size, seq_len] or [seq_len]
        entropy_model: Entropy model
        patcher_args: Patcher configuration (threshold will be overridden to ≥1.35)
        device: Device to run on
    
    Returns:
        Patch lengths in same format as Patcher.patch() [batch_size, num_patches, 1]
    """
    logger.info("=" * 80)
    logger.info("STRICT MONOTONICITY PATCHING")
    logger.info("=" * 80)
    
    # Create patcher with strict settings
    strict_args = patcher_args.model_copy(deep=True)
    strict_args.threshold = max(patcher_args.threshold, 1.35)  # At least 1.35
    strict_args.monotonicity = True
    strict_args.repetition_detection = False
    
    strict_patcher = strict_args.build()
    strict_patcher.entropy_model = entropy_model
    
    # Calculate entropies with chunking for large sequences
    tokens_batch = tokens.unsqueeze(0) if tokens.dim() == 1 else tokens
    seq_len = tokens_batch.shape[1]
    
    if seq_len > 8192:
        logger.info(f"Large sequence detected ({seq_len} bytes) - using chunked entropy calculation")
        chunk_size = 8192
        overlap = 512
        chunks = []
        chunk_starts = []
        start = 0
        while start < seq_len:
            end = min(start + chunk_size, seq_len)
            chunks.append(tokens_batch[:, start:end])
            chunk_starts.append(start)
            start = end - overlap if end < seq_len else end
        
        # Compute entropies chunk by chunk (avoids giant forward passes)
        entropies_list = []
        for chunk_idx, chunk in enumerate(chunks):
            chunk_entropies, _ = calculate_entropies(
                chunk,
                entropy_model,
                patching_batch_size=1,
                device=device,
            )
            entropies_list.append(chunk_entropies)
        
        # Merge entropies with averaging on overlaps
        full_entropies = torch.zeros_like(tokens_batch, dtype=torch.float32)
        overlap_counts = torch.zeros_like(full_entropies)

        for chunk_ent, chunk_start in zip(entropies_list, chunk_starts):
            chunk_end = chunk_start + chunk_ent.shape[1]
            full_entropies[:, chunk_start:chunk_end] += chunk_ent
            overlap_counts[:, chunk_start:chunk_end] += 1
        
        entropies = full_entropies / overlap_counts.clamp(min=1)
    else:
        entropies, _ = calculate_entropies(
            tokens_batch,
            entropy_model,
            patching_batch_size=1,
            device=device,
        )

    # Get patches
    patch_lengths = strict_patcher.patch(
        tokens_batch,
        include_next_token=False,
        entropies=entropies,
    )
    
    logger.info("=" * 80)
    
    return patch_lengths


# Deprecated: Multi-stage functions kept for reference but not used in main pipeline
def three_stage_patch(
    tokens: torch.Tensor,
    entropy_model,
    patcher_args: PatcherArgs,
    device: str = "cpu",
) -> torch.Tensor:
    """
    DEPRECATED: Use strict_monotonicity_patch() instead.
    
    Three-stage patching pipeline (kept for reference only).
    """
    logger.warning("three_stage_patch() is deprecated. Use strict_monotonicity_patch() instead.")
    return strict_monotonicity_patch(tokens, entropy_model, patcher_args, device)

