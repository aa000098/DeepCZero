#!/usr/bin/env python3
"""
Verify Llama 3.2 1B Instruct inference with PyTorch (HuggingFace transformers).
Generates reference output for comparing with DeepCZero C++ implementation.

Usage:
    pip install transformers torch
    python scripts/verify_llama.py "What is the capital of France?"
"""

import sys
import json


def verify(prompt="What is the capital of France?"):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except ImportError:
        print("Error: transformers and torch are required.")
        sys.exit(1)

    model_id = "meta-llama/Llama-3.2-1B-Instruct"

    print(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print(f"Loading model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    model.eval()

    # Apply chat template
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print(f"\nFormatted prompt:\n{repr(formatted)}")

    input_ids = tokenizer.encode(formatted, return_tensors="pt")
    print(f"Input token IDs ({input_ids.shape[1]} tokens): {input_ids[0].tolist()}")

    # Generate
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=64,
            do_sample=False,  # greedy
            temperature=1.0,
        )

    generated_ids = output[0][input_ids.shape[1]:].tolist()
    print(f"\nGenerated token IDs ({len(generated_ids)} tokens): {generated_ids}")

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(f"\nGenerated text: {generated_text}")

    # Also get first-token logits for verification
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits  # [1, seq_len, vocab_size]
        last_logits = logits[0, -1, :]  # [vocab_size]

        # Top-5 tokens
        top5 = torch.topk(last_logits, 5)
        print(f"\nTop-5 next tokens (first generation step):")
        for i in range(5):
            token_id = top5.indices[i].item()
            logit_val = top5.values[i].item()
            token_str = tokenizer.decode([token_id])
            print(f"  [{i}] id={token_id}, logit={logit_val:.4f}, token='{token_str}'")

    # Output reference data as JSON for C++ comparison
    ref_data = {
        "prompt": prompt,
        "formatted": formatted,
        "input_ids": input_ids[0].tolist(),
        "generated_ids": generated_ids,
        "generated_text": generated_text,
        "top5_ids": top5.indices.tolist(),
        "top5_logits": [round(v, 4) for v in top5.values.tolist()],
    }

    ref_path = "scripts/llama_reference.json"
    with open(ref_path, "w") as f:
        json.dump(ref_data, f, indent=2)
    print(f"\nReference data saved to: {ref_path}")


if __name__ == "__main__":
    prompt = sys.argv[1] if len(sys.argv) > 1 else "What is the capital of France?"
    verify(prompt)
