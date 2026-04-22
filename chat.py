"""Interactive multi-turn chat with Gemma 4 E2B-it."""

import argparse

from gemma_chat.model_loading import load_processor_and_model, model_input_device

MODEL_ID = "google/gemma-4-E2B-it"


def load_model(load_mode=None):
    print(f"Loading {MODEL_ID} ...")
    processor, model, load_info = load_processor_and_model(MODEL_ID, load_mode)
    print(f"Model loaded ({load_info.detail}).\n")
    return processor, model


def generate(processor, model, messages, enable_thinking):
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    inputs = processor(text=text, return_tensors="pt").to(model_input_device(model))
    input_len = inputs["input_ids"].shape[-1]

    outputs = model.generate(
        **inputs,
        max_new_tokens=2048,
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        do_sample=True,
    )
    raw = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
    return processor.parse_response(raw)


def main():
    parser = argparse.ArgumentParser(description="Chat with Gemma 4 E2B-it")
    parser.add_argument(
        "--think", action="store_true", help="Enable thinking / reasoning mode"
    )
    parser.add_argument(
        "--load-mode",
        choices=("auto", "bf16", "4bit"),
        default=None,
        help="Model loading mode. auto uses 4-bit on GPUs below 12 GB VRAM.",
    )
    args = parser.parse_args()

    processor, model = load_model(args.load_mode)

    messages = [{"role": "system", "content": "You are a helpful assistant."}]

    print("Type your message ('/think' toggles thinking mode; Ctrl+C exits).")
    print(f"Thinking mode: {'ON' if args.think else 'OFF'}\n")

    enable_thinking = args.think

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() == "/think":
            enable_thinking = not enable_thinking
            print(f"Thinking mode: {'ON' if enable_thinking else 'OFF'}")
            continue
        if user_input.lower() == "/reset":
            messages = [{"role": "system", "content": "You are a helpful assistant."}]
            print("Conversation reset.")
            continue

        messages.append({"role": "user", "content": user_input})

        parsed = generate(processor, model, messages, enable_thinking)

        # Display thinking if present
        if hasattr(parsed, "thinking") and parsed.thinking:
            print(f"\n[Thinking]\n{parsed.thinking}\n")

        response_text = parsed.content if hasattr(parsed, "content") else str(parsed)
        print(f"\nGemma: {response_text}\n")

        # Store only the final response in history (strip thinking per best practices)
        messages.append({"role": "assistant", "content": response_text})


if __name__ == "__main__":
    main()
