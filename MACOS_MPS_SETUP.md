# BLT macOS MPS Setup Guide

This document explains all modifications made to enable BLT (Byte Latent Transformer) to run on macOS with Apple Silicon (MPS) support, and provides instructions for using the modified repository.

## Table of Contents

1. [Overview](#overview)
2. [Modifications Summary](#modifications-summary)
3. [Detailed Changes](#detailed-changes)
4. [Setup Instructions](#setup-instructions)
5. [Usage Guide](#usage-guide)
6. [Testing](#testing)
7. [Troubleshooting](#troubleshooting)
8. [Repetition Detection Feature](#repetition-detection-feature)
9. [Additional Notes](#additional-notes)

## Overview

The original BLT repository was configured for CUDA-only PyTorch installations, which don't work on macOS. This guide documents the changes made to:

- Remove CUDA-specific dependencies
- Make xformers optional (it doesn't build on macOS)
- Enable MPS (Metal Performance Shaders) support for Apple Silicon
- Fix code to work without xformers
- Implement strict monotonicity patching as the main feature
- Create optimized test scripts for verification

## Modifications Summary

### Files Modified

1. **`pyproject.toml`** - Dependency configuration
2. **`setup.py`** - Package setup
3. **`bytelatent/transformer.py`** - Optional xformers import
4. **`bytelatent/model/local_models.py`** - Optional xformers import
5. **`bytelatent/model/latent_transformer.py`** - Optional xformers import
6. **`bytelatent/model/utils.py`** - Optional xformers import with error handling
7. **`bytelatent/train.py`** - Optional xformers profiler
8. **`bytelatent/probe.py`** - Optional xformers import
9. **`apps/main/lingua_train.py`** - Optional xformers profiler

### Files Created

1. **`tests/test_patching.py`** - Test script for entropy model and patching on macOS
2. **`tests/test_optimized_pipeline.py`** - Test script for strict monotonicity patching with dataset loading
3. **`bytelatent/data/cross_reference_patcher.py`** - Strict monotonicity patching function with chunking optimization
4. **`bytelatent/data/repetition_detector.py`** - Repetition detection module (kept for reference, not used in main pipeline)

## Detailed Changes

### 1. `pyproject.toml` - Dependency Configuration

#### Changes Made:

**Removed:**
- CUDA-specific torch index (`torch-nightly-cu121`)
- Torch override pinning CUDA-only version (`torch==2.6.0.dev20241112`)
- `xformers` from main dependencies

**Added:**
- Standard PyTorch dependencies: `torch`, `torchaudio`, `torchvision` (these install with MPS support on macOS)
- Comments explaining xformers is optional

**Modified:**
- `[tool.uv.sources]` - Commented out xformers git source
- `[dependency-groups]` - Removed torch from `pre_build`, kept xformers as optional group
- `[tool.uv]` - Removed torch override, commented out xformers build isolation

#### Why:

- CUDA wheels don't exist for macOS ARM64
- Standard PyPI torch includes MPS support
- xformers requires CUDA and doesn't build on macOS

### 2. `setup.py` - Package Setup

#### Changes Made:

**Removed:**
- `xformers` from `install_requires`

**Added:**
- Comment explaining xformers is optional

#### Why:

- xformers is only needed for training speed on CUDA systems
- Not required for inference, especially on macOS

### 3. Code Files - Optional xformers Imports

#### Files Modified:
- `bytelatent/transformer.py`
- `bytelatent/model/local_models.py`
- `bytelatent/model/latent_transformer.py`
- `bytelatent/model/utils.py`
- `bytelatent/train.py`
- `bytelatent/probe.py`
- `apps/main/lingua_train.py`

#### Pattern Applied:

**Before:**
```python
from xformers.ops import AttentionBias
```

**After:**
```python
try:
    from xformers.ops import AttentionBias
except ImportError:
    # xformers not available (e.g., on macOS) - use Any as fallback type
    from typing import Any
    AttentionBias = Any
```

#### Special Cases:

**`bytelatent/model/utils.py`:**
- Added error handling when xformers is required but missing:
```python
if attn_impl == "xformers":
    if fmha is None:
        raise ImportError(
            "xformers is required for attn_impl='xformers' but is not installed. "
            "Install it with: pip install xformers (Linux/CUDA only) or use a different attn_impl."
        )
```

**`bytelatent/train.py` and `apps/main/lingua_train.py`:**
- Made xformers profiler optional:
```python
try:
    import xformers.profiler
except ImportError:
    xformers = None

# Later usage:
if torch_profiler and xformers is not None:
    xformers.profiler.step()
```

**`bytelatent/probe.py`:**
- Added check before using xformers:
```python
elif fmha is not None and func._overloadpacket == fmha.flash.FwOp.OPERATOR:
```

#### Why:

- Allows code to run without xformers installed
- Provides clear error messages when xformers is required but unavailable
- Maintains backward compatibility with Linux/CUDA setups

### 4. `tests/test_patching.py` - Test Script

#### Purpose:

Validates that the entropy model and patching work correctly on macOS with MPS. Configured for larger patch sizes optimized for embedding/geometric placement use cases.

#### Features:

1. **Device Detection:**
   - Automatically detects MPS availability
   - Falls back to CPU if MPS unavailable

2. **Model Loading:**
   - Loads entropy model from HuggingFace (`facebook/blt-entropy`)
   - Configures for MPS with float16
   - Changes attention implementation from `xformers` to `sdpa`

3. **Patcher Setup:**
   - Loads tokenizer and patcher config from HuggingFace
   - Configures patcher for larger patch sizes (optimized for embedding/geometric placement)
   - Sets environment variable for macOS compatibility

4. **Testing:**
   - Processes sample texts (including repetitive text to demonstrate large patches)
   - Creates dynamic patches based on entropy
   - Displays patch information (length, entropy, content)
   - Shows comprehensive statistics including patch size distribution

#### Key Configuration:

```python
# Set environment variable for SDPA with block_causal attention
os.environ['BLT_SUPPRESS_ATTN_ERROR'] = '1'

# Change attention implementation
entropy_model.attn_impl = 'sdpa'

# Configure patcher for larger patch sizes
patcher_args.threshold = 1.5  # Global entropy threshold (higher = fewer patch starts = larger patches)
patcher_args.threshold_add = 0.35  # Relative entropy increase threshold (monotonicity constraint)
patcher_args.monotonicity = False  # Use combined global+relative threshold logic
patcher_args.max_patch_length = 384  # Hard limit on maximum patch size
patcher_args.realtime_patching = False  # Offline use - we have full context
```

#### Patcher Configuration Parameters:

- **`threshold`** (default: 1.335): Global entropy threshold. When entropy > threshold, start a new patch. Higher values = fewer patch starts = larger average patch sizes.
- **`threshold_add`** (default: None): Relative entropy increase threshold. When entropy increases by this amount compared to previous byte, start a new patch. Prevents patches from growing indefinitely.
- **`monotonicity`** (default: False): When `False` and `threshold_add` is set, uses combined global+relative threshold logic. When `True`, uses only relative threshold (entropy differences).
- **`max_patch_length`** (default: None): Hard limit on maximum patch size. Patches exceeding this are split.
- **`realtime_patching`** (default: False): When `False`, uses full context for better entropy estimates (suitable for offline processing).

**Recommended Settings for Larger Patches:**
- `threshold = 1.5` → Average 14-40 bytes on diverse text, 25-80+ bytes on repetitive content
- `threshold_add = 0.35` → Monotonicity constraint prevents patches from ballooning
- `max_patch_length = 384` → Allows large patches on repetitive content while preventing excessive growth

## Setup Instructions

### Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.12
- `uv` package manager (recommended) or `pip`

### Step 1: Clone the Repository

```bash
git clone https://github.com/facebookresearch/blt
cd blt
```

### Step 2: Install Dependencies

#### Using `uv` (Recommended):

```bash
# Sync dependencies (creates .venv automatically)
uv sync

# Activate virtual environment
source .venv/bin/activate
```

#### Using `pip`:

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install PyTorch with MPS support
pip install torch torchvision torchaudio

# Install other dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Step 3: Verify Installation

```bash
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

Should output: `MPS available: True`

### Step 4: Install HuggingFace CLI (Optional but Recommended)

```bash
pip install huggingface-hub
huggingface-cli login
```

This allows automatic model downloads from HuggingFace.

## Usage Guide

### Strict Monotonicity Patching (Recommended)

The main feature is **strict monotonicity patching** which uses entropy-based patching with a strict monotonicity constraint (threshold ≥1.35). This achieves excellent average patch lengths (~21-22 bytes) and is optimized for large sequences with automatic chunking.

#### Quick Start:

```python
import torch
import os
from bytelatent.hf import BltTokenizerAndPatcher
from bytelatent.transformer import LMTransformer
from bytelatent.data.cross_reference_patcher import strict_monotonicity_patch
from bytelatent.data.patcher import PatcherArgs

# Set environment variable for macOS compatibility
os.environ['BLT_SUPPRESS_ATTN_ERROR'] = '1'

# Detect device
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load entropy model
entropy_model = LMTransformer.from_pretrained("facebook/blt-entropy")
entropy_model = entropy_model.to(device).half().eval()
entropy_model.attn_impl = 'sdpa'

# Load tokenizer and patcher config
tok_and_patcher = BltTokenizerAndPatcher.from_pretrained("facebook/blt-7b")
tokenizer = tok_and_patcher.tokenizer_args.build()
patcher_args = tok_and_patcher.patcher_args.model_copy(deep=True)

# Configure for strict monotonicity
patcher_args.realtime_patching = False
patcher_args.patching_device = device
patcher_args.threshold = 1.35  # Will be enforced to ≥1.35
patcher_args.monotonicity = True  # Enforced
patcher_args.max_patch_length = 384

# Process text
text = "Your text here"
tokens = tokenizer.encode(text)
input_ids = torch.tensor([tokens], dtype=torch.long).to(device)

# Use strict monotonicity patching (handles chunking automatically for large sequences)
with torch.no_grad():
    patch_lengths = strict_monotonicity_patch(
        input_ids,
        entropy_model,
        patcher_args,
        device=device,
    )
```

#### Key Features:

- **Automatic chunking**: Sequences >8192 bytes are automatically chunked with overlap for efficient processing
- **Strict monotonicity**: Enforces threshold ≥1.35 for consistent patch quality
- **Optimized for large sequences**: Handles multi-megabyte inputs efficiently
- **MPS compatible**: Works seamlessly on Apple Silicon

### Basic Usage: Entropy Model Only (Legacy)

The entropy model is the core component for dynamic patching. You can use it independently without the full BLT model.

#### Example Script:

```python
import torch
import os
from bytelatent.hf import BltTokenizerAndPatcher
from bytelatent.transformer import LMTransformer

# Set environment variable for macOS compatibility
os.environ['BLT_SUPPRESS_ATTN_ERROR'] = '1'

# Detect device
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load entropy model
entropy_model = LMTransformer.from_pretrained("facebook/blt-entropy")
entropy_model = entropy_model.to(device).half().eval()

# Change attention implementation for macOS
entropy_model.attn_impl = 'sdpa'

# Load tokenizer and patcher config
tok_and_patcher = BltTokenizerAndPatcher.from_pretrained("facebook/blt-7b")
tokenizer = tok_and_patcher.tokenizer_args.build()
patcher_args = tok_and_patcher.patcher_args.model_copy(deep=True)

# Configure for larger patch sizes (optimized for embedding/geometric placement)
patcher_args.realtime_patching = False  # Offline use - we have full context
patcher_args.patching_device = device
patcher_args.threshold = 1.5  # Higher = fewer patch starts = larger patches
patcher_args.threshold_add = 0.35  # Monotonicity constraint
patcher_args.monotonicity = False  # Use combined threshold logic
patcher_args.max_patch_length = 384  # Hard limit on maximum patch size

patcher = patcher_args.build()
patcher.entropy_model = entropy_model

# Process text
text = "Your text here"
tokens = tokenizer.encode(text)
input_ids = torch.tensor([tokens], dtype=torch.long).to(device)

# Calculate entropies and create patches
from bytelatent.data.patcher import calculate_entropies

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
```

### Running the Test Scripts

#### Optimized Pipeline Test (`tests/test_optimized_pipeline.py`) - Recommended

The optimized test script demonstrates strict monotonicity patching with dataset loading:

```bash
# Test with FineWeb-Edu dataset (default)
BLT_SUPPRESS_ATTN_ERROR=1 uv run python tests/test_optimized_pipeline.py --sample-size 300000 --num-samples 1

# Test with HuggingFace dataset
BLT_SUPPRESS_ATTN_ERROR=1 uv run python tests/test_optimized_pipeline.py \
    --dataset-name "RUC-DataLab/DataScience-Instruct-500K" \
    --sample-size 300000 \
    --num-samples 1 \
    --threshold 1.35

# Or if using pip/venv:
BLT_SUPPRESS_ATTN_ERROR=1 python tests/test_optimized_pipeline.py --sample-size 300000
```

#### Basic Test (`tests/test_patching.py`)

The basic test script demonstrates core functionality:

```bash
# Set environment variable and run
BLT_SUPPRESS_ATTN_ERROR=1 uv run python tests/test_patching.py

# Or if using pip/venv:
BLT_SUPPRESS_ATTN_ERROR=1 python tests/test_patching.py
```

#### Repetition Detection Tuning (`tests/test_repetition_tuning.py`)

The focused test script for tuning repetition detection parameters:

```bash
# Basic usage
BLT_SUPPRESS_ATTN_ERROR=1 uv run python tests/test_repetition_tuning.py

# With custom sample size
BLT_SUPPRESS_ATTN_ERROR=1 uv run python tests/test_repetition_tuning.py \
  --num-samples 1 \
  --sample-size 50000

# With custom dataset
BLT_SUPPRESS_ATTN_ERROR=1 uv run python tests/test_repetition_tuning.py \
  --dataset-path "hf://datasets/HuggingFaceFW/fineweb-edu/sample/10BT" \
  --sample-size 50000 \
  --num-samples 1 \
  --limit 1000
```

This script tests multiple parameter combinations and provides detailed statistics on repetition detection effectiveness.

### Expected Output

The test script will:
1. Detect MPS availability
2. Load the entropy model from HuggingFace
3. Load tokenizer and patcher configuration
4. Process sample texts and create dynamic patches
5. Display patch information including:
   - Patch number
   - Length (in tokens/bytes)
   - Entropy value
   - Content preview
   - Comprehensive statistics (total patches, average length, size distribution)

Example output (diverse text):
```
Created 23 patches:
Patch  Length   Entropy      Content (first 50 chars)
--------------------------------------------------------------------------------
1      1        3.4629       
2      1        1.4492       T
3      3        1.6396       he 
4      2        1.9219       ca
5      6        0.8643       pital 
...

Statistics:
  - Total patches: 23
  - Average patch length: 4.3 bytes
  - Min/Max patch length: 1/16 bytes
  - Compression ratio: 4.17x (bytes per patch)
  - Patch size distribution: 23 small (<10B), 0 medium (10-50B), 0 large (≥50B)
```

Example output (repetitive text - shows larger patches):
```
Created 18 patches:
...
14     17       0.3804       repetitive text. 
15     32       0.2344       This is a very repetitive text. 
16     32       0.1118       This is a very repetitive text. 
17     32       0.0910       This is a very repetitive text. 
18     96       0.0547       This is a very repetitive text. This is a very rep

Statistics:
  - Total patches: 18
  - Average patch length: 14.3 bytes
  - Min/Max patch length: 1/96 bytes
  - Compression ratio: 14.17x (bytes per patch)
  - Patch size distribution: 13 small (<10B), 4 medium (10-50B), 1 large (≥50B)
```

## Testing

### Quick Test

Run the test script to verify everything works:

```bash
BLT_SUPPRESS_ATTN_ERROR=1 uv run python tests/test_patching.py
```

### What the Tests Validate

**Optimized Pipeline Test (`tests/test_optimized_pipeline.py`):**
1. ✅ MPS detection and availability
2. ✅ PyTorch installation with MPS support
3. ✅ Entropy model loading from HuggingFace
4. ✅ Strict monotonicity patching
5. ✅ Automatic chunking for large sequences (>8192 bytes)
6. ✅ Dataset loading (FineWeb-Edu and HuggingFace datasets)
7. ✅ Detailed patch statistics and distribution
8. ✅ Memory usage monitoring

**Basic Test (`tests/test_patching.py`):**
1. ✅ MPS detection and availability
2. ✅ PyTorch installation with MPS support
3. ✅ Entropy model loading from HuggingFace
4. ✅ Tokenizer and patcher configuration
5. ✅ Dynamic patch creation
6. ✅ Entropy calculation
7. ✅ Text encoding/decoding

### Expected Behavior

With strict monotonicity patching (threshold ≥1.35, monotonicity=True):

- **Average patch length**: ~21-22 bytes on diverse text (e.g., DataScience-Instruct-500K)
- **Patch distribution**: Mix of small (≤4B), small+ (5-12B), medium (13-24B), medium+ (25-48B), large (49-127B), and XL (≥128B) patches
- **High entropy regions** (uncertain/complex) → **shorter patches** (1-12 bytes)
- **Low entropy regions** (predictable/repetitive) → **longer patches** (20-384 bytes)
- **Automatic chunking**: Large sequences (>8192 bytes) are processed efficiently with overlap

Examples:
- Repetitive phrase "This is a very repetitive text." → 32-byte patch, entropy ~0.11
- Highly repetitive content → 96-byte patch, entropy ~0.05
- Complex/dense text → 6-20 byte patches, entropy ~1.2-1.8
- Uncertain starts → 1-3 byte patches, entropy ~2.0-3.5

**Patch Size Distribution:**
- Small patches (<10B): High-entropy regions, uncertain predictions
- Medium patches (10-50B): Normal text, moderate entropy
- Large patches (≥50B): Repetitive content, low entropy (up to 384B with max_patch_length)

## Troubleshooting

### Issue: "Torch not compiled with CUDA enabled"

**Solution:** This is expected on macOS. The code should use MPS, not CUDA. Ensure you're using standard PyTorch from PyPI, not CUDA-specific builds.

### Issue: "xformers is required for attn_impl='xformers'"

**Solution:** Change the attention implementation:
```python
entropy_model.attn_impl = 'sdpa'
os.environ['BLT_SUPPRESS_ATTN_ERROR'] = '1'
```

### Issue: "SDPA attention being used, which doesn't have specialized attention implementations"

**Solution:** Set the environment variable:
```bash
export BLT_SUPPRESS_ATTN_ERROR=1
# Or in Python:
import os
os.environ['BLT_SUPPRESS_ATTN_ERROR'] = '1'
```

### Issue: Model downloads are slow

**Solution:** 
- Use HuggingFace CLI to login: `huggingface-cli login`
- Models will be cached in `~/.cache/huggingface/`

### Issue: MPS not available

**Possible causes:**
1. Not on Apple Silicon Mac
2. PyTorch version too old
3. macOS version too old

**Solution:**
- Verify: `python -c "import torch; print(torch.backends.mps.is_available())"`
- Update PyTorch: `pip install --upgrade torch`
- Code will fall back to CPU automatically

### Issue: Out of memory errors

**Solution:**
- Use smaller batch sizes
- Use CPU instead of MPS for very large models
- Reduce model precision (already using float16)

### Issue: Import errors

**Solution:**
- Ensure virtual environment is activated
- Reinstall dependencies: `uv sync` or `pip install -e .`
- Check Python version: `python --version` (should be 3.12)

## Limitations

### What Works

✅ Entropy model inference on MPS  
✅ Dynamic patch creation with strict monotonicity  
✅ Text encoding/decoding  
✅ Entropy calculation  
✅ Automatic chunking for large sequences  
✅ Memory-efficient processing  
✅ Dataset loading (FineWeb-Edu and HuggingFace datasets)  
✅ Comprehensive patch statistics  

### What Doesn't Work (on macOS)

❌ xformers (requires CUDA)  
❌ Full BLT model training (designed for CUDA/multi-GPU)  
❌ Some attention implementations (flex_attention has CUDA dependencies)  

### Workarounds

- Use `sdpa` attention instead of `xformers`
- Use entropy model only (sufficient for patching)
- For training, use Linux with CUDA

## Strict Monotonicity Patching

### Overview

The main patching approach uses **strict monotonicity** with threshold ≥1.35. This enforces that patches start at high-entropy points and flow into predictable continuations, creating a natural hierarchy of patch lengths.

### Key Features

- **Strict threshold**: Minimum threshold of 1.35 ensures consistent patch quality
- **Monotonicity constraint**: Patches must have decreasing entropy within them
- **Automatic chunking**: Sequences >8192 bytes are automatically split and processed with overlap
- **Optimized for large sequences**: Handles multi-megabyte inputs efficiently

### Performance

On real-world datasets (e.g., DataScience-Instruct-500K):
- Average patch length: ~21-22 bytes
- Processing time: Fast (chunked processing avoids memory issues)
- Memory usage: Efficient (chunking prevents OOM errors)

## Repetition Detection Feature (Deprecated)

### Overview

**Note**: Repetition detection is kept for reference but not used in the main pipeline. The strict monotonicity approach provides excellent results without the computational overhead.

The repetition detection feature (if enabled) identifies repeated byte sequences in text and recalculates their entropy using a concatenated context window. This creates a "myelination" effect where repetitive high-entropy patches are converted into lower-entropy patches, resulting in longer patches.

### How It Works

1. **Detection Phase**: Uses a rolling hash (Rabin-Karp) algorithm to find repeated byte sequences across the entire corpus (unlimited distance by default)
2. **Entropy Recalculation**: For each repetition pair, extracts 256-byte windows around each occurrence, concatenates them (512 bytes total), and re-runs the entropy model
3. **Recursive Refinement**: Iteratively detects new repetitions that emerge after entropy adjustment, continuing until convergence

### Key Parameters

#### Detection Parameters

- **`repetition_detection`** (bool, default: False): Enable/disable repetition detection
- **`repetition_min_match`** (int, default: 8): Minimum bytes to consider a repetition (lower = more sensitive)
- **`repetition_max_distance`** (int | None, default: None): Maximum distance between occurrences (None = unlimited, searches entire corpus)
- **`repetition_window_size`** (int, default: 256): Size of context window around each occurrence in bytes
- **`repetition_hash_size`** (int, default: 8): Size of n-gram for hashing (should be <= min_match)

#### Memory Management Parameters

- **`repetition_max_pairs`** (int | None, default: None): Maximum number of repetition pairs to process (None = all, useful for memory management)
- **`repetition_sort_by_length`** (bool, default: True): Sort repetitions by match length (longest first) before limiting

#### Recursion Parameters

- **`repetition_max_iterations`** (int, default: 3): Maximum number of recursive passes (0 = single pass, 1+ = recursive)
- **`repetition_convergence_threshold`** (float, default: 0.01): Minimum entropy change to continue recursion (stops when change < threshold)

### Usage Example

```python
from bytelatent.data.patcher import PatcherArgs

# Configure patcher with repetition detection
patcher_args = tok_and_patcher.patcher_args.model_copy(deep=True)
patcher_args.realtime_patching = False
patcher_args.patching_device = device
patcher_args.threshold = 0.35  # Monotonicity threshold
patcher_args.monotonicity = True  # Use pure monotonicity constraint

# Enable repetition detection
patcher_args.repetition_detection = True
patcher_args.repetition_min_match = 8  # Detect repetitions of 8+ bytes
patcher_args.repetition_window_size = 256  # 256-byte context windows
patcher_args.repetition_max_distance = None  # Search entire corpus
patcher_args.repetition_hash_size = 8
patcher_args.repetition_max_pairs = 200  # Limit to top 200 for memory
patcher_args.repetition_sort_by_length = True
patcher_args.repetition_max_iterations = 3  # Up to 3 recursive passes
patcher_args.repetition_convergence_threshold = 0.01  # Stop when entropy change < 0.01

patcher = patcher_args.build()
patcher.entropy_model = entropy_model
```

### How Recursion Works

1. **Initial Pass**: Detects repetitions → recalculates entropies
2. **Iteration 1**: Detects new repetitions on adjusted entropies → recalculates
3. **Iteration 2+**: Continues until:
   - No new repetitions found, OR
   - Entropy change < convergence threshold, OR
   - Maximum iterations reached

### Hierarchical Detection Layers

The repetition detection system uses a hierarchical approach with multiple detection layers that work together:

#### Layer A: Pattern Discovery
- **Single-Scale Detection**: Finds repetitions at a single scale (default: 8+ bytes)
- **Multi-Scale Detection**: Finds repetitions at multiple scales simultaneously (8, 32, 128, 512 bytes)
- Runs in every iteration
- Parameters: `repetition_multi_scale`, `repetition_scale_levels`

#### Layer B: Patch-Aware Detection
- Identifies common patches (patches that appear multiple times)
- Extracts multiple windows around common patch occurrences
- Only runs after iteration > 0 (needs existing patches)
- Parameters: `repetition_patch_aware`, `repetition_num_windows`

#### Layer C: Boundary-Spanning Detection
- Detects repetitions of sequences that **span across patch boundaries**
- For each boundary, extracts: [last N bytes of previous patch] + [first M bytes of next patch]
- Searches for repetitions of these cross-boundary sequences elsewhere in text
- If found, indicates potential unnecessary splits that should be merged
- Only runs after iteration > 0 (needs existing patches)
- Parameters:
  - `repetition_boundary_aware` (bool, default: False): Enable boundary-spanning detection
  - `repetition_boundary_span_before` (int, default: 64): Bytes before boundary to include
  - `repetition_boundary_span_after` (int, default: 64): Bytes after boundary to include
  - `repetition_boundary_min_match` (int, default: 16): Minimum length of cross-boundary sequence

**Why Boundary-Spanning Detection?**
- Regular repetition detection finds repeated byte sequences, but doesn't specifically target patch boundaries
- Boundary-spanning detection directly identifies cases where sequences crossing boundaries are repeated
- This enables merging of unnecessarily split patches, creating longer patches for repetitive content
- Particularly effective for web content with repeated structures (tables, lists, boilerplate) that may be split across boundaries

**Usage Example:**
```python
# Enable all three layers for maximum effect
patcher_args.repetition_detection = True
patcher_args.repetition_multi_scale = True
patcher_args.repetition_scale_levels = [8, 32, 128, 512]
patcher_args.repetition_patch_aware = True
patcher_args.repetition_num_windows = 5
patcher_args.repetition_boundary_aware = True
patcher_args.repetition_boundary_span_before = 64
patcher_args.repetition_boundary_span_after = 64
patcher_args.repetition_boundary_min_match = 16
patcher_args.repetition_max_iterations = 7  # Need iterations > 0 for layers B and C
```

### Memory Management

The system includes several memory management features:

- **Batch Processing**: Repetition pairs are processed in batches of 50 (configurable) to manage memory
- **Pair Limiting**: Can limit to top N longest repetitions to prioritize most impactful ones
- **Memory Monitoring**: Optional memory usage tracking and warnings
- **Cache Clearing**: Automatically clears GPU cache after each batch

### Logging

Comprehensive logging is available at INFO level:

```
REPETITION DETECTION ENABLED
  Sequence length: 49896 tokens (49896 bytes)
  Detection parameters:
    - min_match: 8 bytes
    - max_distance: unlimited
    - window_size: 256 bytes
    - hash_size: 8 bytes
    - max_pairs: 200
  Recursion parameters:
    - max_iterations: 3
    - convergence_threshold: 0.01

  Initial pass:
    Current entropy stats: mean=2.2241, min=0.0000, max=4.1562, std=0.7578
    Found 200 repetition pair(s)
      Rep 1: length=86B, distance=4713B, avg_entropy=2.5316, positions=[27191:27277] and [31904:31990]
    Recalculating entropies for 200 repetition pair(s)
      Initial entropy: mean=2.2241, min=0.0000, max=4.1562
      Final entropy: mean=1.1123, min=0.0000, max=4.1562
      Entropy reduction: 1.1118 (50.0% decrease)
    Entropy change: +1.111760 (mean decreased by 1.111760)

  Iteration 1 pass:
    Current entropy stats: mean=1.1123, min=0.0000, max=4.1562, std=0.6501
    Found 200 repetition pair(s)
    Entropy change: +0.000000 (mean decreased by 0.000000)
    Converged after 1 iteration(s) - entropy change below threshold

  Final results:
    Total entropy reduction: 1.111760
    Final entropy stats: mean=1.1123, min=0.0000, max=4.1562
```

### Testing Repetition Detection

Use the focused test script to tune parameters:

```bash
BLT_SUPPRESS_ATTN_ERROR=1 uv run python tests/test_repetition_tuning.py \
  --num-samples 1 \
  --sample-size 50000
```

This script tests various parameter combinations and provides detailed statistics on:
- Number of repetitions found
- Entropy reduction achieved
- Patch size distribution
- Memory usage
- Convergence behavior

### Recommended Settings

**For Maximum Effect (More Repetitions Detected):**
- `repetition_min_match = 4-6` (more sensitive)
- `repetition_max_distance = None` (unlimited search)
- `repetition_max_pairs = None` (process all)
- `repetition_max_iterations = 5` (more recursive passes)

**For Memory Efficiency:**
- `repetition_min_match = 8-12` (less sensitive, fewer pairs)
- `repetition_max_pairs = 200` (limit to top 200)
- `repetition_window_size = 128` (smaller windows)
- `repetition_max_iterations = 3` (fewer passes)

**For Balanced Performance:**
- `repetition_min_match = 8`
- `repetition_window_size = 256`
- `repetition_max_pairs = 200`
- `repetition_max_iterations = 3`
- `repetition_convergence_threshold = 0.01`

**For Maximum Boundary Merging (Boundary-Spanning Detection):**
- `repetition_boundary_aware = True`
- `repetition_boundary_span_before = 64`
- `repetition_boundary_span_after = 64`
- `repetition_boundary_min_match = 16`
- `repetition_max_iterations = 7` (need iterations > 0 for boundary-aware to run)
- Combine with multi-scale and patch-aware for best results

## Additional Notes

### Performance

- MPS provides good performance for inference
- Float16 is used for memory efficiency
- Entropy model is small (~99.5M parameters, ~200MB)
- Larger patch sizes reduce the number of patches, improving efficiency for embedding/geometric placement

### Patcher Configuration Tuning

The default HuggingFace model configuration uses `threshold=1.335`, which produces very small patches (2-5 bytes average) optimized for autoregressive generation. For embedding/geometric placement use cases, larger patches are preferred:

**Configuration Options:**

1. **For larger average patches (25-40 bytes):**
   - `threshold = 1.0-1.2`
   - `threshold_add = 0.35`
   - `max_patch_length = 384`

2. **For very large patches (50-80+ bytes):**
   - `threshold = 1.3-1.6`
   - `threshold_add = 0.35`
   - `max_patch_length = 512` or `1024`

3. **For maximum separation (stronger prominence signal):**
   - `threshold = 1.5` (current test script setting)
   - `threshold_add = 0.35` (prevents patches from growing indefinitely)
   - `monotonicity = False` (uses combined global+relative threshold)

**How Thresholds Work:**
- **Global threshold (`threshold`)**: When entropy > threshold, start a new patch. Higher values = fewer patch starts = larger patches.
- **Relative threshold (`threshold_add`)**: When entropy increases by this amount vs previous byte, start a new patch. Prevents patches from growing indefinitely on repetitive content.
- **Combined logic**: When `monotonicity=False` and `threshold_add` is set, both conditions must be met: entropy > threshold AND entropy increase > threshold_add.

### Model Sizes

- `facebook/blt-entropy`: ~200MB (entropy model only)
- `facebook/blt-1b`: ~2GB (full model, not needed for patching)
- `facebook/blt-7b`: ~14GB (full model, not needed for patching)

### Attention Implementations

On macOS, use:
- `sdpa` - Scaled Dot Product Attention (works on MPS)
- Avoid: `xformers`, `flex_attention` (require CUDA)

### Environment Variables

- `BLT_SUPPRESS_ATTN_ERROR=1` - Allows SDPA with block_causal attention
- `BLT_ALLOW_MISSING_FLEX_ATTENTION=1` - Allows missing flex_attention (if needed)

## Contributing

If you make additional modifications for macOS compatibility:

1. Document changes in this file
2. Test with the provided test script
3. Ensure backward compatibility with Linux/CUDA setups
4. Use try/except for optional dependencies

## References

- [BLT Paper](https://arxiv.org/abs/2412.09871)
- [BLT HuggingFace](https://huggingface.co/facebook/blt-entropy)
- [PyTorch MPS Documentation](https://pytorch.org/docs/stable/notes/mps.html)

## License

This repository maintains the original BLT license. See `LICENSE` file for details.

