"""Interactive multi-turn chat with Gemma 4 E2B-it."""

import argparse
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

MODEL_ID = "google/gemma-4-E2B-it"


def load_model():
    print(f"Loading {MODEL_ID} ...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print("Model loaded.\n")
    return processor, model


def generate(processor, model, messages, enable_thinking):
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    inputs = processor(text=text, return_tensors="pt").to(model.device)
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
    args = parser.parse_args()

    processor, model = load_model()

    messages = [{"role": "system", "content": "You are a helpful assistant."}]

    print("Type your message (or 'quit' to exit, '/think' to toggle thinking mode).")
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
        if user_input.lower() == "quit":
            print("Bye!")
            break
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
