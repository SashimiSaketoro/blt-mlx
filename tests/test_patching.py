#!/usr/bin/env python3
"""
Test script to verify entropy model and patching work correctly on macOS with MPS.
This script loads the entropy model and creates dynamic patches from test text.
"""

import torch
from bytelatent.hf import BltTokenizerAndPatcher
from bytelatent.transformer import LMTransformer
from bytelatent.data.patcher import PatcherArgs, to_device, calculate_entropies

def process_text_to_patches(text: str, tokenizer, entropy_model, patcher):
    """
    Process text into patches using the entropy model.
    
    Returns:
        patches_tokens: List of token sequences for each patch
        patch_entropies: List of entropy values for each patch
        patch_lengths: List of patch lengths in tokens
        patches_text: List of decoded text for each patch
    """
    # Encode text using tokenizer (this handles BOS/EOS and offset encoding)
    tokens = tokenizer.encode(text)
    
    # Convert to tensor
    input_ids = torch.tensor([tokens], dtype=torch.long)
    
    # Move to device
    device = next(entropy_model.parameters()).device
    input_ids = input_ids.to(device)
    
    # Calculate entropies for the tokens
    with torch.no_grad():
        entropies, _ = calculate_entropies(
            input_ids,
            entropy_model,
            patching_batch_size=1,
            device=device,
        )
        
        # Get patch lengths using the patcher
        patch_lengths = patcher.patch(
            input_ids,
            include_next_token=False,
            entropies=entropies,
        )
    
    # Extract patch information
    patch_lengths = patch_lengths[0][0].cpu().numpy()  # [max_num_patches]
    
    # Remove zero padding (patches are right-padded with zeros)
    patch_lengths = patch_lengths[patch_lengths > 0]
    
    # Calculate patch starts and extract token sequences
    patch_starts = [0] + patch_lengths.cumsum().tolist()[:-1]
    patches_tokens = [tokens[start:start+length] for start, length in zip(patch_starts, patch_lengths)]
    
    # Decode patches to text using tokenizer
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
    
    return patches_tokens, patch_entropies, patch_lengths.tolist(), patches_text


def main():
    print("=" * 80)
    print("Testing BLT Entropy Model and Patching on macOS")
    print("=" * 80)
    
    # Check MPS availability
    if torch.backends.mps.is_available():
        device = "mps"
        print(f"✓ MPS is available - using Apple Silicon GPU")
    else:
        device = "cpu"
        print(f"⚠ MPS not available - using CPU")
    
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")
    
    print("\n" + "=" * 80)
    print("Loading entropy model from HuggingFace...")
    print("=" * 80)
    
    # Load entropy model
    try:
        entropy_model = LMTransformer.from_pretrained("facebook/blt-entropy")
        
        # Move to device and set dtype
        entropy_model = entropy_model.to(device)
        if device == "mps":
            # MPS works best with float16
            entropy_model = entropy_model.half()
        
        entropy_model = entropy_model.eval()
        # Disable gradients for inference
        for param in entropy_model.parameters():
            param.requires_grad = False
        
        print("✓ Entropy model loaded successfully")
        print(f"  - Model device: {next(entropy_model.parameters()).device}")
        print(f"  - Model dtype: {next(entropy_model.parameters()).dtype}")
    except Exception as e:
        print(f"✗ Failed to load entropy model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 80)
    print("Loading tokenizer and patcher configuration from HuggingFace...")
    print("=" * 80)
    
    # Load tokenizer and patcher config (we need this to get the patcher args)
    try:
        tok_and_patcher = BltTokenizerAndPatcher.from_pretrained("facebook/blt-7b")
        
        # Build tokenizer
        tokenizer = tok_and_patcher.tokenizer_args.build()
        print("✓ Tokenizer loaded successfully")
        
        # Get patcher args
        patcher_args = tok_and_patcher.patcher_args.model_copy(deep=True)
        
        # Configure for larger patch sizes (optimized for embedding/geometric placement)
        # Default HF config produces small patches (2-5 bytes avg) - we want larger (25-40 bytes avg)
        patcher_args.realtime_patching = False  # Offline use - we have full context
        patcher_args.patching_device = device
        patcher_args.threshold = 1.5  # Global entropy threshold - higher = fewer patch starts = larger patches
        patcher_args.threshold_add = 0.35  # Relative entropy increase threshold - when entropy increases by this, start new patch
        # Note: When monotonicity=False and threshold_add is set, uses combined global+relative threshold logic
        patcher_args.monotonicity = False  # Use combined threshold logic (not pure monotonicity)
        patcher_args.max_patch_length = 384  # Hard limit on maximum patch size
        
        # Build patcher
        patcher = patcher_args.build()
        
        # Set the entropy model manually (since we loaded it separately)
        # Change attention implementation to one that works on macOS
        import os
        os.environ['BLT_SUPPRESS_ATTN_ERROR'] = '1'  # Allow SDPA to work with block_causal
        
        if hasattr(entropy_model, 'attn_impl'):
            # Use SDPA which works on MPS (with suppressed error for block_causal)
            entropy_model.attn_impl = 'sdpa'
            print(f"  - Changed attention impl to: sdpa (for macOS compatibility)")
        
        patcher.entropy_model = entropy_model
        
        print("✓ Patcher configured successfully")
        print(f"  - Patching mode: {patcher.patching_mode}")
        print(f"  - Threshold: {patcher.threshold} (higher = fewer patch starts = larger patches)")
        print(f"  - Threshold add (monotonicity): {patcher.threshold_add}")
        print(f"  - Monotonicity enabled: {patcher.monotonicity}")
        print(f"  - Max patch length: {patcher.max_patch_length} bytes")
        print(f"  - Expected avg patch size: 14-40 bytes on diverse text, 25-80+ bytes on repetitive content")
        print(f"  - (vs 2-5 bytes with default threshold=1.335 config)")
    except Exception as e:
        print(f"✗ Failed to load tokenizer/patcher config: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 80)
    print("Testing with sample texts...")
    print("=" * 80)
    
    # Test texts - including longer, more repetitive text to test larger patches
    test_texts = [
        "The capital of France is Paris. Paris is in France. The quick brown fox jumps over the lazy dog.",
        "Hello world! This is a test of the entropy-based patching system.",
        "Machine learning is fascinating. Neural networks can learn complex patterns from data.",
        "This is a very repetitive text. This is a very repetitive text. This is a very repetitive text. This is a very repetitive text. This is a very repetitive text. This is a very repetitive text. This is a very repetitive text. This is a very repetitive text.",
        "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog.",
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n--- Test {i} ---")
        print(f"Text: {text[:60]}...")
        print(f"Length: {len(text)} bytes")
        
        try:
            patches_tokens, patch_entropies, patch_lengths, patches_text = process_text_to_patches(
                text, tokenizer, entropy_model, patcher
            )
            
            print(f"\nCreated {len(patches_text)} patches:")
            print(f"{'Patch':<6} {'Length':<8} {'Entropy':<12} {'Content (first 50 chars)'}")
            print("-" * 80)
            
            for j, (patch_text, entropy, length) in enumerate(zip(patches_text, patch_entropies, patch_lengths), 1):
                preview = patch_text[:50].replace('\n', '\\n')
                print(f"{j:<6} {length:<8} {entropy:<12.4f} {preview}")
            
            # Statistics
            avg_entropy = sum(patch_entropies) / len(patch_entropies)
            avg_length = sum(patch_lengths) / len(patch_lengths)
            min_length = min(patch_lengths)
            max_length = max(patch_lengths)
            total_bytes = sum(patch_lengths)
            compression_ratio = len(text) / len(patches_text) if len(patches_text) > 0 else 0
            
            print(f"\nStatistics:")
            print(f"  - Total patches: {len(patches_text)}")
            print(f"  - Average patch length: {avg_length:.1f} bytes")
            print(f"  - Min/Max patch length: {min_length}/{max_length} bytes")
            print(f"  - Average entropy: {avg_entropy:.4f}")
            print(f"  - Entropy range: {min(patch_entropies):.4f} - {max(patch_entropies):.4f}")
            print(f"  - Compression ratio: {compression_ratio:.2f}x (bytes per patch)")
            
            # Show patch size distribution
            small_patches = sum(1 for l in patch_lengths if l < 10)
            medium_patches = sum(1 for l in patch_lengths if 10 <= l < 50)
            large_patches = sum(1 for l in patch_lengths if l >= 50)
            print(f"  - Patch size distribution: {small_patches} small (<10B), {medium_patches} medium (10-50B), {large_patches} large (≥50B)")
            
        except Exception as e:
            print(f"✗ Error processing text: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("✓ All tests completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()

