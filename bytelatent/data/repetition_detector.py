# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Repetition detection module for identifying repeated text patterns.
Uses rolling hash (Rabin-Karp) algorithm for efficient substring matching.
"""

import logging
from collections import defaultdict
from typing import List, Tuple

import torch

logger = logging.getLogger(__name__)


def detect_repetitions(
    tokens: torch.Tensor,
    min_match_length: int = 8,
    max_distance: int | None = None,
    window_size: int = 256,
    hash_size: int = 8,
    max_repetitions: int | None = None,
    sort_by_length: bool = True,
) -> List[Tuple[int, int, int, int]]:
    """
    Detect repeated text patterns using rolling hash algorithm.
    
    Args:
        tokens: 1D tensor of byte tokens [seq_len] or 2D tensor [batch_size, seq_len]
        min_match_length: Minimum number of bytes to consider a repetition (default: 8, lower = more sensitive)
        max_distance: Maximum distance between occurrences to consider (None = unlimited, can search entire corpus)
        window_size: Size of context window around each occurrence (default: 256)
        hash_size: Size of n-gram for hashing (default: 8, should be <= min_match_length)
        max_repetitions: Maximum number of repetition pairs to return (None = all, useful for memory management)
        sort_by_length: If True, sort repetitions by match length (longest first) before limiting
    
    Returns:
        List of tuples (start1, end1, start2, end2) representing repetition pairs.
        Each tuple indicates two occurrences of the same text pattern.
    """
    # Handle 2D tensors (batch_size, seq_len) - process first batch item
    if tokens.dim() == 2:
        tokens = tokens[0]
    
    seq_len = tokens.shape[0]
    if seq_len < min_match_length * 2:
        logger.debug(f"Sequence too short for repetition detection: {seq_len} < {min_match_length * 2}")
        return []
    
    max_dist_str = "unlimited" if max_distance is None else str(max_distance)
    logger.info(f"    Starting repetition detection:")
    logger.info(f"      Sequence length: {seq_len} bytes")
    logger.info(f"      Min match length: {min_match_length} bytes")
    logger.info(f"      Max distance: {max_dist_str}")
    logger.info(f"      Hash size: {hash_size} bytes")
    logger.info(f"      Max repetitions: {'all' if max_repetitions is None else max_repetitions}")
    
    # Convert to numpy for efficient byte comparison
    tokens_np = tokens.cpu().numpy()
    
    # Use rolling hash to find potential matches
    # Hash table: hash_value -> list of (start_position, end_position)
    hash_table = defaultdict(list)
    
    # Prime for rolling hash (using a large prime)
    prime = 1000000007
    base = 256
    
    # Precompute base powers for rolling hash
    base_powers = [1]
    for i in range(1, hash_size):
        base_powers.append((base_powers[-1] * base) % prime)
    
    # Compute rolling hash for all n-grams of size hash_size
    matches = []
    
    # First pass: compute hashes and find collisions
    for i in range(seq_len - hash_size + 1):
        # Compute hash for window [i:i+hash_size]
        hash_value = 0
        for j in range(hash_size):
            hash_value = (hash_value + tokens_np[i + j] * base_powers[j]) % prime
        
        # Store position with hash
        hash_table[hash_value].append(i)
    
    # Second pass: for hash collisions, verify actual matches and extend them
    for hash_value, positions in hash_table.items():
        if len(positions) < 2:
            continue
        
        # Check all pairs of positions with this hash
        for idx1, pos1 in enumerate(positions):
            for pos2 in positions[idx1 + 1:]:
                # Check distance constraint (if specified)
                if max_distance is not None:
                    distance = abs(pos2 - pos1)
                    if distance > max_distance:
                        continue
                
                # Verify actual match (avoid false positives from hash collision)
                if pos1 + min_match_length > seq_len or pos2 + min_match_length > seq_len:
                    continue
                
                # Compare actual bytes
                match_len = 0
                max_extend = min(seq_len - pos1, seq_len - pos2, min_match_length * 4)
                
                for k in range(max_extend):
                    if tokens_np[pos1 + k] == tokens_np[pos2 + k]:
                        match_len += 1
                    else:
                        break
                
                # If match is long enough, record it
                if match_len >= min_match_length:
                    # Extend match backwards if possible
                    start1 = pos1
                    start2 = pos2
                    extend_back = min(pos1, pos2, window_size // 2)
                    
                    for k in range(1, extend_back + 1):
                        if tokens_np[pos1 - k] == tokens_np[pos2 - k]:
                            start1 = pos1 - k
                            start2 = pos2 - k
                        else:
                            break
                    
                    # Extend match forwards
                    end1 = pos1 + match_len
                    end2 = pos2 + match_len
                    extend_forward = min(seq_len - end1, seq_len - end2, window_size // 2)
                    
                    for k in range(extend_forward):
                        # Check bounds before accessing
                        if end1 + k >= seq_len or end2 + k >= seq_len:
                            break
                        if tokens_np[end1 + k] == tokens_np[end2 + k]:
                            end1 += 1
                            end2 += 1
                        else:
                            break
                    
                    # Ensure we have valid windows
                    match_length = end1 - start1
                    if match_length >= min_match_length:
                        matches.append((start1, end1, start2, end2))
    
    # Remove overlapping matches (keep longest)
    matches = _deduplicate_matches(matches)
    
    # Sort by length if requested (longest first)
    if sort_by_length and matches:
        matches_with_length = [(abs(e1 - s1), s1, e1, s2, e2) for s1, e1, s2, e2 in matches]
        matches_with_length.sort(reverse=True)
        matches = [(s1, e1, s2, e2) for _, s1, e1, s2, e2 in matches_with_length]
    
    # Limit number of repetitions if specified
    if max_repetitions is not None and len(matches) > max_repetitions:
        logger.info(f"Limiting repetitions from {len(matches)} to {max_repetitions} (keeping longest)")
        matches = matches[:max_repetitions]
    
    if matches:
        logger.info(f"Found {len(matches)} repetition pair(s):")
        for idx, (s1, e1, s2, e2) in enumerate(matches[:5], 1):  # Log first 5
            match_len = e1 - s1
            distance = abs(s2 - s1)
            logger.info(f"  Repetition {idx}: length={match_len}B, distance={distance}B, "
                       f"positions=[{s1}:{e1}] and [{s2}:{e2}]")
        if len(matches) > 5:
            logger.info(f"  ... and {len(matches) - 5} more repetition(s)")
    else:
        logger.info("No repetitions detected")
    
    return matches


def detect_repetitions_multi_scale(
    tokens: torch.Tensor,
    scale_levels: list[int] = [8, 32, 128, 512],
    max_distance: int | None = None,
    window_size: int = 256,
    max_repetitions: int | None = None,
    sort_by_length: bool = True,
) -> List[Tuple[int, int, int, int]]:
    """
    Detect repetitions at multiple scales simultaneously.
    
    Args:
        tokens: 1D tensor of byte tokens [seq_len] or 2D tensor [batch_size, seq_len]
        scale_levels: List of scales (min_match_length values) to detect at
        max_distance: Maximum distance between occurrences (None = unlimited)
        window_size: Size of context window around each occurrence
        max_repetitions: Maximum number of repetition pairs to return per scale
        sort_by_length: If True, sort repetitions by match length (longest first)
    
    Returns:
        Merged list of repetition pairs from all scales, deduplicated
    """
    if tokens.dim() == 2:
        tokens = tokens[0]
    
    seq_len = tokens.shape[0]
    all_matches = []
    
    logger.info(f"Multi-scale detection: scales={scale_levels}, seq_len={seq_len}")
    
    # Detect at each scale
    for scale in scale_levels:
        if scale > seq_len // 2:
            logger.debug(f"Skipping scale {scale} (too large for sequence length {seq_len})")
            continue
        
        hash_size = min(scale, 8)  # Hash size should be <= min_match
        matches = detect_repetitions(
            tokens,
            min_match_length=scale,
            max_distance=max_distance,
            window_size=window_size,
            hash_size=hash_size,
            max_repetitions=max_repetitions,
            sort_by_length=sort_by_length,
        )
        
        logger.info(f"  Scale {scale}B: found {len(matches)} repetition pair(s)")
        all_matches.extend(matches)
    
    # Deduplicate matches (keep longest ones)
    if all_matches:
        all_matches = _deduplicate_matches(all_matches)
        logger.info(f"Multi-scale detection: {len(all_matches)} total unique repetition pair(s) after deduplication")
    
    return all_matches


def identify_common_patches(
    patch_lengths: torch.Tensor,
    tokens: torch.Tensor,
    min_occurrences: int = 3,
    min_patch_length: int = 8,
) -> List[Tuple[int, int, List[int]]]:
    """
    Identify common patches (patches that appear multiple times).
    
    Args:
        patch_lengths: Tensor of patch lengths [num_patches]
        tokens: 1D tensor of byte tokens [seq_len] or 2D tensor [batch_size, seq_len]
        min_occurrences: Minimum number of times a patch must appear to be considered common
        min_patch_length: Minimum patch length to consider
    
    Returns:
        List of (start, end, occurrence_positions) tuples for common patches
    """
    if tokens.dim() == 2:
        tokens = tokens[0]
    
    tokens_np = tokens.cpu().numpy()
    patch_lengths_np = patch_lengths.cpu().numpy() if isinstance(patch_lengths, torch.Tensor) else patch_lengths
    
    # Build patch positions
    patch_starts = []
    current_pos = 0
    for length in patch_lengths_np:
        if length > 0:
            patch_starts.append(current_pos)
            current_pos += int(length)
    
    # Extract patches and find duplicates
    patch_contents = {}
    for i, start in enumerate(patch_starts):
        length = int(patch_lengths_np[i])
        if length < min_patch_length:
            continue
        
        end = start + length
        if end > len(tokens_np):
            continue
        
        patch_bytes = bytes(tokens_np[start:end])
        if patch_bytes not in patch_contents:
            patch_contents[patch_bytes] = []
        patch_contents[patch_bytes].append((start, end))
    
    # Find common patches
    common_patches = []
    for patch_bytes, occurrences in patch_contents.items():
        if len(occurrences) >= min_occurrences:
            # Use first occurrence as reference
            start, end = occurrences[0]
            occurrence_positions = [start for start, _ in occurrences]
            common_patches.append((start, end, occurrence_positions))
    
    logger.info(f"Identified {len(common_patches)} common patch(es) (appearing {min_occurrences}+ times)")
    return common_patches


def extract_multi_windows_around_patches(
    tokens: torch.Tensor,
    common_patches: List[Tuple[int, int, List[int]]],
    num_windows: int = 5,
    window_size: int = 32,
) -> List[Tuple[int, int, int, int]]:
    """
    Extract multiple windows around common patch occurrences for entropy recalculation.
    
    Args:
        tokens: 1D tensor of byte tokens [seq_len] or 2D tensor [batch_size, seq_len]
        common_patches: List of (start, end, occurrence_positions) tuples
        num_windows: Number of windows to extract around each occurrence
        window_size: Size of each window in bytes
    
    Returns:
        List of (start1, end1, start2, end2) tuples representing window pairs
    """
    if tokens.dim() == 2:
        tokens = tokens[0]
    
    seq_len = tokens.shape[0]
    window_pairs = []
    
    for patch_start, patch_end, occurrence_positions in common_patches:
        if len(occurrence_positions) < 2:
            continue
        
        patch_length = patch_end - patch_start
        
        # For each pair of occurrences, extract multiple windows
        for i in range(len(occurrence_positions)):
            for j in range(i + 1, len(occurrence_positions)):
                pos1 = occurrence_positions[i]
                pos2 = occurrence_positions[j]
                
                # Extract num_windows windows around each occurrence
                # Windows are centered around the patch occurrence
                patch_center1 = pos1 + patch_length // 2
                patch_center2 = pos2 + patch_length // 2
                
                # Calculate window positions (staggered around the patch)
                for win_idx in range(num_windows):
                    # Offset from patch center (stagger windows around the patch)
                    offset = (win_idx - num_windows // 2) * window_size
                    
                    # Calculate window 1
                    win_center1 = patch_center1 + offset
                    win_start1 = max(0, win_center1 - window_size // 2)
                    win_end1 = min(seq_len, win_start1 + window_size)
                    # Adjust if we hit boundaries
                    if win_end1 - win_start1 < window_size:
                        if win_start1 == 0:
                            win_end1 = min(seq_len, window_size)
                        else:
                            win_start1 = max(0, win_end1 - window_size)
                    
                    # Calculate window 2
                    win_center2 = patch_center2 + offset
                    win_start2 = max(0, win_center2 - window_size // 2)
                    win_end2 = min(seq_len, win_start2 + window_size)
                    # Adjust if we hit boundaries
                    if win_end2 - win_start2 < window_size:
                        if win_start2 == 0:
                            win_end2 = min(seq_len, window_size)
                        else:
                            win_start2 = max(0, win_end2 - window_size)
                    
                    # Ensure windows are valid and of exact size
                    if win_end1 - win_start1 == window_size and win_end2 - win_start2 == window_size:
                        if win_end1 <= seq_len and win_end2 <= seq_len:
                            window_pairs.append((win_start1, win_end1, win_start2, win_end2))
    
    logger.info(f"Extracted {len(window_pairs)} window pair(s) around {len(common_patches)} common patch(es)")
    return window_pairs


def detect_boundary_spanning_repetitions(
    tokens: torch.Tensor,
    patch_start_ids: torch.Tensor,
    span_before: int = 64,
    span_after: int = 64,
    min_match_length: int = 16,
    max_distance: int | None = None,
    max_repetitions: int | None = None,
    hash_size: int = 8,
) -> List[Tuple[int, int, int, int]]:
    """
    Detect repetitions of byte sequences that span across patch boundaries.
    
    For each patch boundary, extracts a cross-boundary sequence consisting of:
    - Last span_before bytes of the previous patch
    - First span_after bytes of the next patch
    
    Then searches for repetitions of these cross-boundary sequences elsewhere in the text.
    If found, these indicate potential unnecessary splits that should be merged.
    
    Args:
        tokens: 1D tensor of byte tokens [seq_len] or 2D tensor [batch_size, seq_len]
        patch_start_ids: Patch start positions [batch_size, num_patches] or 1D [num_patches]
        span_before: Number of bytes before boundary to include (default: 64)
        span_after: Number of bytes after boundary to include (default: 64)
        min_match_length: Minimum length of cross-boundary sequence to consider (default: 16)
        max_distance: Maximum distance between occurrences to consider (None = unlimited)
        max_repetitions: Maximum number of repetitions to return (None = all)
        hash_size: Size of n-gram for hashing (default: 8, should be <= min_match_length)
    
    Returns:
        List of (start1, end1, start2, end2) tuples for cross-boundary repetition pairs
    """
    if tokens.dim() == 2:
        tokens = tokens[0]
    
    if patch_start_ids.dim() == 2:
        patch_start_ids = patch_start_ids[0]
    
    seq_len = tokens.shape[0]
    
    # Convert patch_start_ids to indices if it's a boolean mask
    if patch_start_ids.dtype == torch.bool:
        patch_start_indices = torch.where(patch_start_ids)[0].cpu().numpy()
    else:
        patch_start_indices = patch_start_ids.cpu().numpy()
    
    # Filter out invalid patch starts (beyond sequence length)
    patch_start_indices = [idx for idx in patch_start_indices if 0 <= idx < seq_len]
    
    if len(patch_start_indices) < 2:
        logger.debug("Not enough patch boundaries for boundary-spanning detection")
        return []
    
    # Sort patch starts
    patch_start_indices = sorted(patch_start_indices)
    
    logger.info(f"    Starting boundary-spanning repetition detection:")
    logger.info(f"      Sequence length: {seq_len} bytes")
    logger.info(f"      Number of patch boundaries: {len(patch_start_indices)}")
    logger.info(f"      Span before boundary: {span_before} bytes")
    logger.info(f"      Span after boundary: {span_after} bytes")
    logger.info(f"      Min match length: {min_match_length} bytes")
    
    tokens_np = tokens.cpu().numpy()
    
    # Extract cross-boundary sequences for each boundary
    cross_boundary_sequences = []
    boundary_positions = []
    
    for i in range(1, len(patch_start_indices)):  # Skip first boundary (start of sequence)
        boundary_pos = patch_start_indices[i]
        prev_patch_start = patch_start_indices[i - 1]
        
        # Calculate previous patch end (current boundary - 1)
        prev_patch_end = boundary_pos - 1
        
        # Extract: [last span_before bytes of previous patch] + [first span_after bytes of next patch]
        seq_start = max(prev_patch_start, prev_patch_end - span_before + 1)
        seq_end = min(seq_len, boundary_pos + span_after)
        
        # Ensure we have a valid sequence
        if seq_start < seq_end and seq_end - seq_start >= min_match_length:
            cross_boundary_seq = tokens_np[seq_start:seq_end]
            cross_boundary_sequences.append((seq_start, seq_end, cross_boundary_seq))
            boundary_positions.append(boundary_pos)
    
    if not cross_boundary_sequences:
        logger.debug("No valid cross-boundary sequences extracted")
        return []
    
    logger.info(f"      Extracted {len(cross_boundary_sequences)} cross-boundary sequence(s)")
    
    # Log sample cross-boundary sequences for debugging
    if cross_boundary_sequences:
        sample_seq = cross_boundary_sequences[0][2]
        sample_bytes = bytes(sample_seq[:min(32, len(sample_seq))])
        logger.info(f"      Sample cross-boundary sequence (first 32B): {sample_bytes!r}")
        logger.info(f"      Sample sequence length: {len(sample_seq)} bytes")
        logger.info(f"      Sample sequence span: [{cross_boundary_sequences[0][0]}:{cross_boundary_sequences[0][1]}]")
        if len(cross_boundary_sequences) > 1:
            logger.info(f"      ... and {len(cross_boundary_sequences) - 1} more cross-boundary sequence(s)")
    
    # Build hash table for entire sequence (O(n) preprocessing, like detect_repetitions)
    # This allows O(1) lookup per cross-boundary sequence instead of O(n) scan
    matches = []
    
    # Prime for rolling hash
    prime = 1000000007
    base = 256
    
    # Precompute base powers for rolling hash
    base_powers = [1]
    for i in range(1, hash_size):
        base_powers.append((base_powers[-1] * base) % prime)
    
    # Build hash table for all positions in sequence
    hash_table = defaultdict(list)
    for i in range(seq_len - hash_size + 1):
        hash_value = 0
        for j in range(hash_size):
            hash_value = (hash_value + tokens_np[i + j] * base_powers[j]) % prime
        hash_table[hash_value].append(i)
    
    logger.debug(f"      Built hash table with {len(hash_table)} unique hash values")
    
    # For each cross-boundary sequence, find repetitions using hash table lookup
    for seq_idx, (seq_start, seq_end, cross_boundary_seq) in enumerate(cross_boundary_sequences):
        seq_len_actual = len(cross_boundary_seq)
        
        if seq_len_actual < min_match_length:
            continue
        
        # Compute hash for the cross-boundary sequence (use first hash_size bytes)
        seq_hash = 0
        for j in range(min(hash_size, seq_len_actual)):
            seq_hash = (seq_hash + cross_boundary_seq[j] * base_powers[j]) % prime
        
        # Look up candidate positions in hash table (O(1) lookup)
        candidate_positions = hash_table.get(seq_hash, [])
        
        # Verify actual matches at candidate positions
        for search_pos in candidate_positions:
            # Skip if this is the original position
            if search_pos == seq_start:
                continue
            
            # Verify actual byte match
            match_len = 0
            max_extend = min(seq_len_actual, seq_len - search_pos)
            
            for k in range(max_extend):
                if search_pos + k >= seq_len:
                    break
                if tokens_np[search_pos + k] == cross_boundary_seq[k]:
                    match_len += 1
                else:
                    break
            
            # If match is long enough, record it
            if match_len >= min_match_length:
                # Check distance constraint if specified
                if max_distance is not None:
                    distance = abs(search_pos - seq_start)
                    if distance > max_distance:
                        continue
                
                # Extend match backwards if possible
                start1 = seq_start
                start2 = search_pos
                extend_back = min(seq_start, search_pos, span_before + span_after)
                
                for k in range(1, extend_back + 1):
                    if seq_start - k < 0 or search_pos - k < 0:
                        break
                    if tokens_np[seq_start - k] == tokens_np[search_pos - k]:
                        start1 = seq_start - k
                        start2 = search_pos - k
                    else:
                        break
                
                # Extend match forwards
                end1 = seq_start + match_len
                end2 = search_pos + match_len
                extend_forward = min(seq_len - end1, seq_len - end2, span_before + span_after)
                
                for k in range(extend_forward):
                    if end1 + k >= seq_len or end2 + k >= seq_len:
                        break
                    if tokens_np[end1 + k] == tokens_np[end2 + k]:
                        end1 += 1
                        end2 += 1
                    else:
                        break
                
                # Ensure we have valid match
                match_length = end1 - start1
                if match_length >= min_match_length:
                    matches.append((start1, end1, start2, end2))
    
    # Remove overlapping matches (keep longest)
    matches = _deduplicate_matches(matches)
    
    # Sort by length if needed (longest first)
    if matches:
        matches_with_length = [(abs(e1 - s1), s1, e1, s2, e2) for s1, e1, s2, e2 in matches]
        matches_with_length.sort(reverse=True)
        matches = [(s1, e1, s2, e2) for _, s1, e1, s2, e2 in matches_with_length]
    
    # Limit number of repetitions if specified
    if max_repetitions is not None and len(matches) > max_repetitions:
        logger.info(f"Limiting boundary-spanning repetitions from {len(matches)} to {max_repetitions} (keeping longest)")
        matches = matches[:max_repetitions]
    
    if matches:
        logger.info(f"      Found {len(matches)} boundary-spanning repetition pair(s):")
        for idx, (s1, e1, s2, e2) in enumerate(matches[:3], 1):
            match_len = e1 - s1
            distance = abs(s2 - s1)
            logger.info(f"        Boundary-span {idx}: length={match_len}B, distance={distance}B, "
                       f"positions=[{s1}:{e1}] and [{s2}:{e2}]")
        if len(matches) > 3:
            logger.info(f"        ... and {len(matches) - 3} more boundary-spanning repetition(s)")
    else:
        logger.info("      No boundary-spanning repetitions detected")
    
    return matches


def _deduplicate_matches(matches: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
    """
    Remove overlapping matches, keeping the longest ones.
    
    Args:
        matches: List of (start1, end1, start2, end2) tuples
    
    Returns:
        Deduplicated list of matches
    """
    if not matches:
        return []
    
    # Sort by match length (longest first)
    matches_with_length = [(abs(e1 - s1), s1, e1, s2, e2) for s1, e1, s2, e2 in matches]
    matches_with_length.sort(reverse=True)
    
    # Keep non-overlapping matches
    kept = []
    used_positions1 = set()
    used_positions2 = set()
    
    for length, s1, e1, s2, e2 in matches_with_length:
        # Check if this match overlaps with any kept match
        overlap1 = any(s1 < end and e1 > start for start, end in used_positions1)
        overlap2 = any(s2 < end and e2 > start for start, end in used_positions2)
        
        if not overlap1 and not overlap2:
            kept.append((s1, e1, s2, e2))
            used_positions1.add((s1, e1))
            used_positions2.add((s2, e2))
    
    return kept

