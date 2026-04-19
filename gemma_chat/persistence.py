"""Conversation, settings, and transcript persistence mixin."""

import json
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox

from .config import APP_FOLDER_NAME, MODEL_ID
from .storage import conversation_path, read_settings, system_prompt_path, write_settings


class PersistenceMixin:
    def _setup_logging(self):
        self.log_dir = self._resolve_log_dir()
        if not self.log_dir:
            self.status_var.set("Logging disabled: no transcript folder selected.")
            return

        self._load_saved_system_prompt()
        self._save_system_prompt()
        self._start_new_log()

    def _read_settings(self) -> dict:
        return read_settings()

    def _write_settings(self, settings: dict):
        write_settings(settings)

    def _load_saved_system_prompt(self):
        saved_prompt = None
        prompt_path = self._system_prompt_path()
        if prompt_path and prompt_path.exists():
            try:
                saved_prompt = prompt_path.read_text(encoding="utf-8")
            except OSError:
                saved_prompt = None

        settings = self._read_settings()
        old_saved_prompt = settings.get("system_prompt")
        if saved_prompt is None and isinstance(old_saved_prompt, str):
            saved_prompt = old_saved_prompt

        if "system_prompt" in settings:
            settings.pop("system_prompt", None)
            self._write_settings(settings)

        if not isinstance(saved_prompt, str):
            return

        self.system_prompt.delete("1.0", tk.END)
        self.system_prompt.insert("1.0", saved_prompt)
        self._fit_system_prompt_to_content()
        self.system_prompt.edit_modified(False)

    def _on_system_prompt_modified(self, _event=None):
        if not self.system_prompt.edit_modified():
            return

        self.system_prompt.edit_modified(False)
        if self._system_prompt_save_job:
            self.root.after_cancel(self._system_prompt_save_job)
        self._system_prompt_save_job = self.root.after(500, self._save_system_prompt)

    def _save_system_prompt(self):
        self._system_prompt_save_job = None
        prompt_path = self._system_prompt_path()
        if not prompt_path:
            return

        try:
            prompt_path.write_text(self._get_system_prompt(), encoding="utf-8")
        except OSError as exc:
            self.status_var.set(f"System prompt save failed: {exc}")

    def _system_prompt_path(self) -> Path | None:
        if not self.log_dir:
            return None

        return system_prompt_path(self.log_dir)

    def _conversation_path(self) -> Path | None:
        if not self.log_dir:
            return None

        return conversation_path(self.log_dir)

    def _load_saved_conversation(self):
        path = self._conversation_path()
        if not path or not path.exists():
            self.messages = []
            return

        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self.messages = []
            return

        if not isinstance(loaded, list):
            self.messages = []
            return

        messages: list[dict] = []
        for item in loaded:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            content = item.get("content")
            if role in {"user", "assistant"} and isinstance(content, str):
                message = {"role": role, "content": content}
                if not messages or messages[-1] != message:
                    messages.append(message)

        self.messages = messages
        self._render_saved_conversation()

    def _save_conversation(self):
        path = self._conversation_path()
        if not path:
            return

        compacted: list[dict] = []
        for message in self.messages:
            if compacted and compacted[-1] == message:
                continue
            compacted.append(message)
        if len(compacted) != len(self.messages):
            self.messages = compacted

        try:
            path.write_text(
                json.dumps(self.messages, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as exc:
            self.status_var.set(f"Conversation save failed: {exc}")

    def _render_saved_conversation(self):
        if not self.messages:
            return

        self._append_chat("System: ", "system_msg")
        self._append_chat("Restored previous conversation.\n\n", "system_msg")
        for message in self.messages:
            role = message["role"]
            content = message["content"]
            if role == "user":
                self._append_chat("You: ", "user")
                self._append_chat(f"{content}\n\n")
            elif role == "assistant":
                self._append_chat("Gemma: ", "assistant")
                self._append_markdown(content)
                self._append_chat("\n\n")

    def _resolve_log_dir(self) -> Path | None:
        configured = self._read_configured_log_dir()
        if configured:
            return configured

        for candidate in (Path.home() / APP_FOLDER_NAME, Path.cwd() / APP_FOLDER_NAME):
            if candidate.exists() and candidate.is_dir():
                self._save_configured_log_dir(candidate)
                return candidate

        parent = filedialog.askdirectory(
            parent=self.root,
            title=f"Choose where to create {APP_FOLDER_NAME}",
            mustexist=True,
        )
        if not parent:
            messagebox.showwarning(
                "Conversation Logging",
                "No transcript folder was selected. Conversation logging is disabled for this session.",
                parent=self.root,
            )
            return None

        log_dir = Path(parent) / APP_FOLDER_NAME
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            messagebox.showerror(
                "Conversation Logging",
                f"Could not create transcript folder:\n{log_dir}\n\n{exc}",
                parent=self.root,
            )
            return None

        self._save_configured_log_dir(log_dir)
        return log_dir

    def _read_configured_log_dir(self) -> Path | None:
        raw_path = self._read_settings().get("log_dir")
        if not raw_path:
            return None

        path = Path(raw_path).expanduser()
        return path if path.exists() and path.is_dir() else None

    def _save_configured_log_dir(self, log_dir: Path):
        settings = self._read_settings()
        settings["log_dir"] = str(log_dir.resolve())
        self._write_settings(settings)

    def _start_new_log(self):
        if not self.log_dir:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = self.log_dir / f"gemma4_chat_{timestamp}.md"
        counter = 2
        while self.log_path.exists():
            self.log_path = self.log_dir / f"gemma4_chat_{timestamp}_{counter}.md"
            counter += 1
        header = (
            "# Gemma 4 E2B-it Chat Log\n\n"
            f"Started: {datetime.now().isoformat(timespec='seconds')}\n\n"
            f"Model: `{MODEL_ID}`\n\n"
            "## Assistant Behaviour\n\n"
            f"{self._get_system_prompt() or '(empty)'}\n\n"
            "---\n\n"
        )
        self._append_log_text(header)

    def _append_log_text(self, text: str):
        if not self.log_path:
            return

        try:
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(text)
        except OSError as exc:
            self.log_path = None
            self.status_var.set(f"Logging disabled: {exc}")

    def _append_log_entry(self, role: str, content: str):
        if not content:
            return

        timestamp = datetime.now().isoformat(timespec="seconds")
        self._append_log_text(f"## {role} - {timestamp}\n\n{content.strip()}\n\n")

    def _append_generation_settings_log(self, enable_thinking: bool):
        settings = (
            f"Thinking Mode: {'on' if enable_thinking else 'off'}\n\n"
            f"Temperature: {self.temp_var.get()}\n\n"
            f"Top-p: {self.top_p_var.get()}\n\n"
            f"Top-k: {self.top_k_var.get()}\n\n"
            f"Max tokens: {self.max_tokens_var.get()}\n\n"
            f"Assistant Behaviour:\n\n{self._get_system_prompt() or '(empty)'}"
        )
        self._append_log_entry("Generation Settings", settings)

