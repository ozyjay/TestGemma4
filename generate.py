"""Single-shot text generation with Gemma 4 E2B-it."""

import argparse
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

MODEL_ID = "google/gemma-4-E2B-it"


def main():
    parser = argparse.ArgumentParser(description="Generate text with Gemma 4 E2B-it")
    parser.add_argument("prompt", help="The prompt to send to the model")
    parser.add_argument(
        "--think", action="store_true", help="Enable thinking / reasoning mode"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=2048, help="Max new tokens to generate"
    )
    parser.add_argument(
        "--system", default="You are a helpful assistant.", help="System prompt"
    )
    args = parser.parse_args()

    print(f"Loading {MODEL_ID} ...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    messages = [
        {"role": "system", "content": args.system},
        {"role": "user", "content": args.prompt},
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=args.think,
    )
    inputs = processor(text=text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    outputs = model.generate(
        **inputs,
        max_new_tokens=args.max_tokens,
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        do_sample=True,
    )
    raw = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
    parsed = processor.parse_response(raw)

    if hasattr(parsed, "thinking") and parsed.thinking:
        print(f"\n[Thinking]\n{parsed.thinking}\n")

    response_text = parsed.content if hasattr(parsed, "content") else str(parsed)
    print(f"\n{response_text}")


if __name__ == "__main__":
    main()
