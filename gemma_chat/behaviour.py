"""Assistant behaviour editing and rewrite mixin."""

import re
import sys
import threading
import traceback
import tkinter as tk

import torch

from .model import _split_gemma_channels


class BehaviourMixin:
    def _reset_conversation(self):
        self.messages = []

    def _get_system_prompt(self) -> str:
        return self.system_prompt.get("1.0", tk.END).strip()

    def _fit_system_prompt_to_content(self):
        line_count = int(self.system_prompt.index("end-1c").split(".")[0])
        self.system_prompt_lines.set(line_count)
        self._apply_system_prompt_height()

    def _apply_system_prompt_height(self):
        try:
            requested = int(self.system_prompt_lines.get())
        except (tk.TclError, ValueError):
            requested = self._system_prompt_min_lines

        height = max(self._system_prompt_min_lines, min(self._system_prompt_max_lines, requested))
        try:
            current = int(self.system_prompt_lines.get())
        except (tk.TclError, ValueError):
            current = None
        if current != height:
            self.system_prompt_lines.set(height)
        if int(self.system_prompt.cget("height")) != height:
            self.system_prompt.configure(height=height)

    def _set_system_prompt(self, text: str):
        self.system_prompt.delete("1.0", tk.END)
        self.system_prompt.insert("1.0", text)
        self._fit_system_prompt_to_content()
        self.system_prompt.see(tk.END)
        self.system_prompt.edit_modified(False)
        self._save_system_prompt()

    def _start_behaviour_rewrite(self, advice: str):
        advice = advice.strip()
        if not advice:
            self.status_var.set("Type a behaviour change first.")
            return

        if self.model is None or self.processor is None:
            self.status_var.set("Wait for the model to finish loading before updating behaviour.")
            return

        if self.generating:
            self.status_var.set("Stop the current generation before updating behaviour.")
            return

        if self.updating_behaviour:
            self.status_var.set("Behaviour update already in progress.")
            return

        self.updating_behaviour = True
        self.behaviour_btn.configure(state=tk.DISABLED)
        self.send_btn.configure(state=tk.DISABLED)
        self.status_var.set("Rewriting assistant behaviour...")
        self._append_chat("System: ", "system_msg")
        self._append_chat(f"Rewriting assistant behaviour from advice: {advice}\n\n", "system_msg")
        self._append_log_entry("Behaviour Rewrite Advice", advice)
        self._stream_thinking_text = ""
        self._begin_thinking_block()
        current_behaviour = self._get_system_prompt()
        threading.Thread(
            target=self._rewrite_behaviour,
            args=(current_behaviour, advice),
            daemon=True,
        ).start()

    def _rewrite_behaviour(self, current_behaviour: str, advice: str):
        rewrite_messages = [
            {
                "role": "system",
                "content": (
                    "You are a careful prompt editor. Rewrite assistant behaviour "
                    "instructions for a chat application. Preserve useful existing "
                    "requirements, incorporate the user's requested change, remove "
                    "contradictions and obsolete wording, and return only the complete "
                    "replacement behaviour instructions. Do not explain your changes."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Current assistant behaviour instructions:\n"
                    "```text\n"
                    f"{current_behaviour or 'You are a helpful assistant.'}\n"
                    "```\n\n"
                    "Requested behaviour change:\n"
                    "```text\n"
                    f"{advice}\n"
                    "```\n\n"
                    "Return the complete replacement Assistant Behaviour text only."
                ),
            },
        ]

        try:
            prompt = self.processor.apply_chat_template(
                rewrite_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            inputs = self.processor(text=prompt, return_tensors="pt").to(self.model.device)
            input_length = inputs["input_ids"].shape[-1]
            with torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=False,
                )
            new_tokens = output_ids[:, input_length:]
            raw_text = self.processor.tokenizer.decode(
                new_tokens[0],
                skip_special_tokens=False,
            )
            thinking_text, response_text = _split_gemma_channels(raw_text)
            if thinking_text:
                thinking_text = thinking_text.replace("\\n", "\n")
            rewritten = self._clean_behaviour_rewrite(response_text or raw_text)
            if not rewritten:
                raise ValueError("The behaviour rewrite was empty.")

            self.root.after(
                0,
                lambda: self._finish_behaviour_rewrite(advice, rewritten, thinking_text),
            )
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            self.root.after(0, lambda err=exc: self._fail_behaviour_rewrite(err))

    def _clean_behaviour_rewrite(self, text: str) -> str:
        cleaned = text.replace("\\n", "\n").strip()
        cleaned = re.sub(r"<\|?channel\|?>\s*\w*|<\w+\|>|<\|turn\>|<turn\|>", "", cleaned)
        cleaned = cleaned.strip()

        fence_match = re.fullmatch(r"```(?:text|markdown|md)?\s*\n?(.*?)\n?```", cleaned, re.DOTALL)
        if fence_match:
            cleaned = fence_match.group(1).strip()

        return cleaned

    def _finish_behaviour_rewrite(self, advice: str, rewritten: str, thinking_text: str | None):
        if thinking_text:
            self._stream_thinking_text = thinking_text
            self._replace_streamed_thinking_with_markdown(thinking_text)
            self._append_log_entry("Thinking", thinking_text)
        else:
            self._discard_active_thinking_block()
        self._set_system_prompt(rewritten)
        self._append_chat("System: ", "system_msg")
        self._append_chat(
            "Assistant behaviour rewritten for future responses.\n\n",
            "system_msg",
        )
        self._append_log_entry("Behaviour Rewrite", f"Advice:\n{advice}\n\nUpdated behaviour:\n{rewritten}")
        self.status_var.set("Assistant behaviour rewritten for future responses.")
        self.updating_behaviour = False
        self.behaviour_btn.configure(state=tk.NORMAL)
        self.send_btn.configure(state=tk.NORMAL)

    def _fail_behaviour_rewrite(self, error: Exception):
        self._discard_active_thinking_block()
        self._append_chat("System: ", "system_msg")
        self._append_chat(f"Behaviour rewrite failed: {error}\n\n", "system_msg")
        self._append_log_entry("Behaviour Rewrite Error", str(error))
        self.status_var.set("Behaviour rewrite failed.")
        self.updating_behaviour = False
        self.behaviour_btn.configure(state=tk.NORMAL)
        self.send_btn.configure(state=tk.NORMAL)

    def _extract_behaviour_command(self, text: str) -> str | None:
        stripped = text.strip()
        lowered = stripped.lower()
        for command in ("/behaviour", "/behavior"):
            if lowered == command:
                return ""
            if lowered.startswith(command + " "):
                return stripped[len(command):].strip()
        return None

    def _highlight_user_input_commands(self):
        self.user_input.tag_remove("slash_command", "1.0", tk.END)
        self.user_input.tag_remove("slash_command_arg", "1.0", tk.END)

        text = self.user_input.get("1.0", "end-1c")
        match = re.match(r"^(/(?:behaviour|behavior|think|reset))(\s+.*)?$", text, re.IGNORECASE | re.DOTALL)
        if not match:
            return

        command_end = len(match.group(1))
        self.user_input.tag_add("slash_command", "1.0", f"1.0+{command_end}c")
        if match.group(2):
            self.user_input.tag_add("slash_command_arg", f"1.0+{command_end}c", tk.END)

    def _on_user_input_changed(self, _event=None):
        self.root.after_idle(self._highlight_user_input_commands)

    def _build_messages(self) -> list[dict]:
        self._save_system_prompt()
        return [{"role": "system", "content": self._get_system_prompt()}] + self.messages

