"""Single-shot text generation with Gemma 4 E2B-it."""

import argparse

from gemma_chat.model_loading import load_processor_and_model, model_input_device

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
    parser.add_argument(
        "--load-mode",
        choices=("auto", "bf16", "4bit"),
        default=None,
        help="Model loading mode. auto uses 4-bit on GPUs below 12 GB VRAM.",
    )
    args = parser.parse_args()

    print(f"Loading {MODEL_ID} ...")
    processor, model, load_info = load_processor_and_model(MODEL_ID, args.load_mode)
    print(f"Model loaded ({load_info.detail}).")

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
    inputs = processor(text=text, return_tensors="pt").to(model_input_device(model))
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
