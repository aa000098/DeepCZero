#!/usr/bin/env python3
"""
Convert Llama 3.2 1B Instruct weights from HuggingFace safetensors (bfloat16)
to numpy NPZ (float32) for DeepCZero framework.

Usage:
    pip install safetensors torch transformers numpy
    python scripts/convert_llama_weights.py meta-llama/Llama-3.2-1B-Instruct
    python scripts/convert_llama_weights.py /path/to/local/Llama-3.2-1B-Instruct

Output: ~/.deepczero/weights/llama-3.2-1b-instruct.npz

Note: For gated models, you need to authenticate:
    huggingface-cli login
"""

import sys
import os
import numpy as np


def convert_llama(model_id, output_path=None):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except ImportError:
        print("Error: transformers and torch are required.")
        print("Install with: pip install transformers torch safetensors")
        sys.exit(1)

    # Default output path
    if output_path is None:
        cache_dir = os.path.join(os.path.expanduser("~"), ".deepczero", "weights")
        os.makedirs(cache_dir, exist_ok=True)
        output_path = os.path.join(cache_dir, "llama-3.2-1b-instruct.npz")

    # Save tokenizer.json
    tokenizer_out = os.path.join(os.path.dirname(output_path), "tokenizer.json")
    if not os.path.exists(tokenizer_out):
        print(f"Saving tokenizer to: {tokenizer_out}")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.save_pretrained(os.path.dirname(output_path))
        print(f"Tokenizer saved: {tokenizer_out}")
    else:
        print(f"Tokenizer already exists: {tokenizer_out}")

    print(f"Loading model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map="cpu",
    )

    sd = model.state_dict()
    print(f"State dict: {len(sd)} keys")

    out = {}
    skipped = 0

    for key, tensor in sd.items():
        arr = tensor.cpu().float().numpy()

        # Determine the DeepCZero key name
        npz_key = key

        # Skip keys we don't need
        if "rotary_emb" in key:
            skipped += 1
            continue

        # Embedding: model.embed_tokens.weight -> model.embed_tokens.W
        # Keep shape [vocab_size, hidden_size] as-is for gather_rows lookup
        if key == "model.embed_tokens.weight":
            npz_key = "model.embed_tokens.W"
            # No transpose needed - gather_rows selects rows
            out[npz_key] = arr
            print(f"  {key:60s} -> {npz_key:60s}  shape={list(arr.shape)}")
            continue

        # lm_head.weight: skip if tied (same as embed_tokens)
        if key == "lm_head.weight":
            # Check if it's actually tied
            embed_w = sd.get("model.embed_tokens.weight")
            if embed_w is not None and torch.equal(tensor, embed_w):
                print(f"  {key:60s} -> SKIPPED (tied with embed_tokens)")
                skipped += 1
                continue
            else:
                npz_key = "lm_head.W"
                # Transpose for Linear: [out, in] -> [in, out]
                arr = arr.T
                out[npz_key] = arr
                print(f"  {key:60s} -> {npz_key:60s}  shape={list(arr.shape)} (transposed)")
                continue

        # Linear weight: *.{q,k,v,o,gate,up,down}_proj.weight -> *.W
        # DeepCZero Linear does x @ W, HF stores [out_features, in_features]
        # So we transpose: [out, in] -> [in, out]
        if key.endswith("_proj.weight"):
            npz_key = key.replace(".weight", ".W")
            arr = arr.T  # Transpose for DeepCZero convention
            out[npz_key] = arr
            print(f"  {key:60s} -> {npz_key:60s}  shape={list(arr.shape)} (transposed)")
            continue

        # RMSNorm weight: stays as-is (1D), keep .weight suffix
        if "layernorm.weight" in key or key == "model.norm.weight":
            npz_key = key  # Keep original name
            out[npz_key] = arr
            print(f"  {key:60s} -> {npz_key:60s}  shape={list(arr.shape)}")
            continue

        # Anything else: store as-is
        out[npz_key] = arr
        print(f"  {key:60s} -> {npz_key:60s}  shape={list(arr.shape)}")

    print(f"\nSaving {len(out)} tensors (skipped {skipped}) to {output_path}")
    np.savez(output_path, **out)
    file_size = os.path.getsize(output_path)
    print(f"File size: {file_size / 1024 / 1024:.1f} MB")

    # Verify
    print("\nVerification:")
    loaded = np.load(output_path)
    print(f"  Keys in npz: {len(loaded.files)}")

    expected = [
        ("model.embed_tokens.W", (128256, 2048)),
        ("model.layers.0.self_attn.q_proj.W", (2048, 2048)),
        ("model.layers.0.self_attn.k_proj.W", (2048, 512)),
        ("model.layers.0.self_attn.v_proj.W", (2048, 512)),
        ("model.layers.0.self_attn.o_proj.W", (2048, 2048)),
        ("model.layers.0.mlp.gate_proj.W", (2048, 8192)),
        ("model.layers.0.mlp.up_proj.W", (2048, 8192)),
        ("model.layers.0.mlp.down_proj.W", (8192, 2048)),
        ("model.layers.0.input_layernorm.weight", (2048,)),
        ("model.layers.0.post_attention_layernorm.weight", (2048,)),
        ("model.norm.weight", (2048,)),
    ]

    all_ok = True
    for name, expected_shape in expected:
        if name in loaded:
            actual_shape = loaded[name].shape
            status = "OK" if actual_shape == expected_shape else f"SHAPE MISMATCH (got {actual_shape})"
            if actual_shape != expected_shape:
                all_ok = False
            print(f"  {status}: {name} shape={actual_shape}")
        else:
            print(f"  MISSING: {name}")
            all_ok = False

    if all_ok:
        print("\nAll checks passed!")
    else:
        print("\nSome checks failed!")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/convert_llama_weights.py <model_id_or_path> [output_path]")
        print("Example: python scripts/convert_llama_weights.py meta-llama/Llama-3.2-1B-Instruct")
        sys.exit(1)

    model_id = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else None
    convert_llama(model_id, out)
