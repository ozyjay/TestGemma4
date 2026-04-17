"""Desktop chat UI for Gemma 4 E2B-it using tkinter with modern UX."""

import json
import queue
import re
import sys
import threading
import time
import traceback
import tkinter as tk
import tkinter.font as tkfont
from datetime import datetime
from pathlib import Path
from tkinter import ttk, scrolledtext, filedialog, messagebox

import psutil
import sv_ttk
import torch
from huggingface_hub import snapshot_download
from markdown_it import MarkdownIt
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, StoppingCriteria, TextIteratorStreamer


# ---------------------------------------------------------------------------
# Custom stopping criteria — allows interrupting generation from the UI
# ---------------------------------------------------------------------------
class _StopOnEvent(StoppingCriteria):
    """Tells model.generate() to stop when a threading.Event is set."""

    def __init__(self, event: threading.Event):
        self._event = event

    def __call__(self, input_ids, scores, **kwargs):
        return self._event.is_set()


def _split_gemma_channels(raw_text: str) -> tuple[str | None, str]:
    """Split Gemma thinking-mode output into thinking and final response."""
    text = raw_text or ""
    text = text.replace("<|turn>", "").replace("<turn|>", "")
    channel_re = re.compile(
        r"<\|channel\|?>\s*(thought|thinking|analysis|response|final|answer)|"
        r"<(thought|thinking|analysis|response|final|answer)\|>|"
        r"<channel\|>",
        re.IGNORECASE,
    )

    matches = list(channel_re.finditer(text))
    if not matches:
        return None, text.strip()

    thinking_parts: list[str] = []
    response_parts: list[str] = []
    current = "response"

    for index, match in enumerate(matches):
        channel = (match.group(1) or match.group(2) or "").lower()
        if channel in {"thought", "thinking", "analysis"}:
            current = "thinking"
        elif channel in {"response", "final", "answer"} or match.group(0) == "<channel|>":
            current = "response"

        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        segment = text[start:end]
        if current == "thinking":
            thinking_parts.append(segment)
        else:
            response_parts.append(segment)

    return "".join(thinking_parts).strip() or None, "".join(response_parts).strip()

try:
    import pynvml

    pynvml.nvmlInit()
    _NVML_AVAILABLE = True
except Exception:
    _NVML_AVAILABLE = False

MODEL_ID = "google/gemma-4-E2B-it"
APP_FOLDER_NAME = ".test.gemma4"
SYSTEM_PROMPT_FILE_NAME = "system_prompt.md"
SETTINGS_DIR = Path.home() / "AppData" / "Roaming" / "TestGemma4"
SETTINGS_FILE = SETTINGS_DIR / "settings.json"


class TkLogStream:
    """Tee stdout/stderr to the original stream and the app diagnostics sink."""

    def __init__(self, app: "GemmaChat", original_stream, tag: str):
        self.app = app
        self.original_stream = original_stream
        self.tag = tag

    def write(self, text):
        if not text:
            return 0

        if self.original_stream:
            try:
                self.original_stream.write(text)
                self.original_stream.flush()
            except Exception:
                pass

        try:
            self.app._capture_diagnostic(text, self.tag)
        except Exception:
            pass

        return len(text)

    def flush(self):
        if self.original_stream:
            try:
                self.original_stream.flush()
            except Exception:
                pass

    def isatty(self):
        if self.original_stream and hasattr(self.original_stream, "isatty"):
            try:
                return self.original_stream.isatty()
            except Exception:
                return False
        return False

    def __getattr__(self, name):
        if self.original_stream:
            return getattr(self.original_stream, name)
        raise AttributeError(name)

# ---------------------------------------------------------------------------
# Color palettes
# ---------------------------------------------------------------------------
THEMES = {
    "dark": {
        "text_bg": "#1e1e1e",
        "text_fg": "#d4d4d4",
        "insert_bg": "#d4d4d4",
        "select_bg": "#264f78",
        "user": "#569cd6",
        "assistant": "#6a9955",
        "thinking": "#c586c0",
        "system_msg": "#808080",
        "bold": "#e0e0e0",
        "italic": "#c8c8c8",
        "heading": "#dcdcaa",
        "code_fg": "#ce9178",
        "code_bg": "#2d2d2d",
        "blockquote": "#808080",
        "stats_fg": "#888888",
    },
    "light": {
        "text_bg": "#ffffff",
        "text_fg": "#1e1e1e",
        "insert_bg": "#1e1e1e",
        "select_bg": "#add6ff",
        "user": "#2563eb",
        "assistant": "#16a34a",
        "thinking": "#9333ea",
        "system_msg": "#6b7280",
        "bold": "#1e1e1e",
        "italic": "#333333",
        "heading": "#b5651d",
        "code_fg": "#c7254e",
        "code_bg": "#f3f4f6",
        "blockquote": "#6b7280",
        "stats_fg": "#555555",
    },
}


# ---------------------------------------------------------------------------
# Markdown renderer
# ---------------------------------------------------------------------------
class MarkdownRenderer:
    """Inserts markdown-formatted text into a tk.Text widget using tags."""

    _md = MarkdownIt("commonmark")

    @classmethod
    def render(cls, widget: tk.Text, text: str):
        """Parse *text* as markdown and insert styled content into *widget*."""
        tokens = cls._md.parse(text)
        cls._walk(widget, tokens)

    @classmethod
    def _walk(cls, widget: tk.Text, tokens):
        tag_stack: list[str] = []
        list_stack: list[dict[str, int | bool]] = []
        in_list_item = False

        for tok in tokens:
            # --- block openers / closers ---
            if tok.type == "heading_open":
                level = int(tok.tag[1])  # h1..h6
                tag_stack.append(f"h{level}")
            elif tok.type == "heading_close":
                widget.insert(tk.END, "\n\n")
                if tag_stack:
                    tag_stack.pop()
            elif tok.type == "paragraph_open":
                pass
            elif tok.type == "paragraph_close":
                # Inside list items, use single newline; otherwise double
                if in_list_item:
                    widget.insert(tk.END, "\n")
                else:
                    widget.insert(tk.END, "\n\n")
            elif tok.type == "blockquote_open":
                tag_stack.append("blockquote")
            elif tok.type == "blockquote_close":
                if tag_stack and tag_stack[-1] == "blockquote":
                    tag_stack.pop()
            elif tok.type == "bullet_list_open":
                list_stack.append({"ordered": False, "counter": 0})
            elif tok.type == "bullet_list_close":
                if list_stack:
                    list_stack.pop()
                if not list_stack:
                    widget.insert(tk.END, "\n")
            elif tok.type == "ordered_list_open":
                start = 1
                if tok.attrs:
                    for key, value in tok.attrs.items():
                        if key == "start":
                            try:
                                start = int(value)
                            except (TypeError, ValueError):
                                start = 1
                            break
                list_stack.append({"ordered": True, "counter": start})
            elif tok.type == "ordered_list_close":
                if list_stack:
                    list_stack.pop()
                if not list_stack:
                    widget.insert(tk.END, "\n")
            elif tok.type == "list_item_open":
                in_list_item = True
                depth = max(0, len(list_stack) - 1)
                indent = "  " * depth
                prefix = "• "
                if list_stack and bool(list_stack[-1]["ordered"]):
                    counter = int(list_stack[-1]["counter"])
                    prefix = f"{counter}. "
                    list_stack[-1]["counter"] = counter + 1
                widget.insert(
                    tk.END,
                    f"{indent}{prefix}",
                    tuple(tag_stack) if tag_stack else (),
                )
            elif tok.type == "list_item_close":
                in_list_item = False
            elif tok.type == "fence":
                # Show language label if present
                info = (tok.info or "").strip()
                if info:
                    widget.insert(tk.END, f"[{info}]\n", ("code_block",))
                widget.insert(tk.END, tok.content, ("code_block",))
                widget.insert(tk.END, "\n")
            elif tok.type == "code_block":
                widget.insert(tk.END, tok.content, ("code_block",))
                widget.insert(tk.END, "\n")
            elif tok.type == "hr":
                widget.insert(tk.END, "─" * 40 + "\n\n")
            elif tok.type == "inline":
                cls._render_inline(widget, tok.children or [], tag_stack)

    @classmethod
    def _render_inline(cls, widget: tk.Text, children, parent_tags: list[str]):
        tag_stack = list(parent_tags)
        for tok in children:
            if tok.type == "text":
                widget.insert(tk.END, tok.content, tuple(tag_stack) if tag_stack else ())
            elif tok.type == "softbreak":
                widget.insert(tk.END, "\n", tuple(tag_stack) if tag_stack else ())
            elif tok.type == "hardbreak":
                widget.insert(tk.END, "\n", tuple(tag_stack) if tag_stack else ())
            elif tok.type == "strong_open":
                tag_stack.append("bold")
            elif tok.type == "strong_close":
                if "bold" in tag_stack:
                    tag_stack.remove("bold")
            elif tok.type == "em_open":
                tag_stack.append("italic")
            elif tok.type == "em_close":
                if "italic" in tag_stack:
                    tag_stack.remove("italic")
            elif tok.type == "code_inline":
                widget.insert(tk.END, tok.content, ("code_inline",))
            elif tok.type == "html_inline":
                widget.insert(tk.END, tok.content, tuple(tag_stack) if tag_stack else ())


# ---------------------------------------------------------------------------
# Stats monitor
# ---------------------------------------------------------------------------
class StatsMonitor:
    def __init__(self):
        self._gpu_handle = None
        if _NVML_AVAILABLE:
            try:
                self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception:
                pass
        # Prime cpu_percent so the first real call returns meaningful data
        psutil.cpu_percent(interval=None)

    def get(self) -> str:
        parts = []
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        parts.append(f"CPU: {cpu:.0f}%")
        parts.append(f"RAM: {mem.used / 1073741824:.1f}/{mem.total / 1073741824:.1f} GB")

        if self._gpu_handle:
            try:
                mi = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
                temp = pynvml.nvmlDeviceGetTemperature(
                    self._gpu_handle, pynvml.NVML_TEMPERATURE_GPU
                )
                parts.append(
                    f"GPU: {mi.used / 1073741824:.1f}/{mi.total / 1073741824:.1f} GB"
                )
                parts.append(f"GPU Util: {util.gpu}%")
                parts.append(f"Temp: {temp}°C")
            except Exception:
                parts.append("GPU: N/A")

        return "  |  ".join(parts)


# ---------------------------------------------------------------------------
# HuggingFace download progress bar → tkinter
# ---------------------------------------------------------------------------
class TkProgressBar(tqdm):
    """Custom tqdm that forwards progress to tkinter variables."""

    _tk_progress_var: tk.DoubleVar | None = None
    _tk_status_var: tk.StringVar | None = None
    _tk_root: tk.Tk | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update(self, n=1):
        super().update(n)
        if self.total and self._tk_progress_var and self._tk_root:
            pct = (self.n / self.total) * 100
            desc = self.desc or "Downloading"
            self._tk_root.after(
                0,
                lambda: (
                    self._tk_progress_var.set(pct),
                    self._tk_status_var.set(
                        f"{desc}: {self.n / 1048576:.0f}/{self.total / 1048576:.0f} MB"
                    )
                    if self._tk_status_var
                    else None,
                ),
            )

    def close(self):
        super().close()


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------
class GemmaChat:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Gemma 4 E2B-it Chat")
        self.root.geometry("960x750")
        self.root.minsize(700, 500)

        self.processor = None
        self.model = None
        self.messages: list[dict] = []
        self.generating = False
        self.updating_behaviour = False
        self._pending_send = False
        self._stop_event = threading.Event()
        self._pending_steer: str | None = None
        self.stats = StatsMonitor()
        self._load_start: float = 0
        self._stream_response_text = ""
        self._stream_thinking_text = ""
        self._stream_response_pending_newline = False
        self._response_render_job: str | None = None
        self._thinking_render_job: str | None = None
        self._has_thinking_history = False
        self._active_thinking_block = False
        self.log_dir: Path | None = None
        self.log_path: Path | None = None
        self._system_prompt_save_job: str | None = None
        self.diagnostics_log_path: Path | None = None
        self._diagnostics_log_handle = None
        self._diagnostics_visible = False
        self._loading_screen_visible = False
        self._diagnostics_queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self._diagnostics_flush_job: str | None = None
        self._stdout_original = sys.stdout
        self._stderr_original = sys.stderr
        self._diagnostics_redirected = False

        # Font settings
        self._available_fonts: list[str] = []
        self.font_family = tk.StringVar(value="Cascadia Code")
        self.font_size = tk.IntVar(value=11)

        # Theme
        self.is_dark = True
        sv_ttk.set_theme("dark")

        self._build_ui()
        self._apply_theme()
        self._apply_fonts()
        self._reset_conversation()
        self._setup_logging()
        self._setup_diagnostics_capture()
        self.system_prompt.bind("<<Modified>>", self._on_system_prompt_modified)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._start_stats_loop()
        self._load_model_async()

    # ── UI construction ─────────────────────────────────────────────────

    def _build_ui(self):
        # --- Toolbar ---
        toolbar = ttk.Frame(self.root, padding=(8, 6))
        toolbar.pack(fill=tk.X)

        # Theme toggle
        self.theme_btn = ttk.Button(
            toolbar, text="☀ Light", width=8, command=self._toggle_theme
        )
        self.theme_btn.pack(side=tk.LEFT, padx=(0, 10))

        # Font family
        ttk.Label(toolbar, text="Font:").pack(side=tk.LEFT)
        all_system = {f for f in tkfont.families() if not f.startswith("@")}
        coding_fonts = [
            "Cascadia Code", "Cascadia Mono",
            "Consolas", "Courier New",
            "Fira Code", "Fira Mono",
            "Hack", "Inconsolata",
            "JetBrains Mono", "Menlo", "Monaco",
            "Source Code Pro", "Ubuntu Mono",
            "DejaVu Sans Mono", "Droid Sans Mono",
            "IBM Plex Mono", "Roboto Mono",
            "SF Mono", "Victor Mono",
            "Anonymous Pro", "Input Mono",
            "Iosevka", "Noto Sans Mono",
            "PT Mono", "Space Mono",
        ]
        available = sorted(
            [f for f in coding_fonts if f in all_system], key=str.lower
        )
        if not available:
            # Fallback: include any mono/courier fonts from the system
            available = sorted(
                [f for f in all_system if any(k in f.lower() for k in ("mono", "courier", "consol"))],
                key=str.lower,
            ) or ["Courier New"]
        self._available_fonts = available
        # Resolve default font
        for preferred in ("Cascadia Code", "Cascadia Mono", "Consolas", "Courier New"):
            if preferred in available:
                self.font_family.set(preferred)
                break
        font_combo = ttk.Combobox(
            toolbar,
            textvariable=self.font_family,
            values=available,
            width=20,
            state="readonly",
        )
        font_combo.pack(side=tk.LEFT, padx=(2, 10))
        font_combo.bind("<<ComboboxSelected>>", lambda _: self._apply_fonts())

        # Font size
        ttk.Label(toolbar, text="Size:").pack(side=tk.LEFT)
        size_spin = ttk.Spinbox(
            toolbar,
            from_=8,
            to=24,
            increment=1,
            textvariable=self.font_size,
            width=3,
            command=self._apply_fonts,
        )
        size_spin.pack(side=tk.LEFT, padx=(2, 10))

        # Thinking toggle
        self.think_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            toolbar, text="Thinking Mode", variable=self.think_var
        ).pack(side=tk.LEFT, padx=(0, 10))

        self.diagnostics_btn = ttk.Button(
            toolbar, text="Diagnostics", command=self._toggle_diagnostics_panel
        )
        self.diagnostics_btn.pack(side=tk.LEFT, padx=(0, 10))

        # --- Assistant behaviour ---
        sys_frame = ttk.LabelFrame(self.root, text="Assistant Behaviour", padding=4)
        sys_frame.pack(fill=tk.X, padx=8, pady=(0, 4))

        self.system_prompt = tk.Text(sys_frame, height=2, wrap=tk.WORD, relief=tk.FLAT)
        self.system_prompt.insert("1.0", "You are a helpful assistant.")
        self.system_prompt.pack(fill=tk.X)

        # --- Generation params (collapsible row) ---
        params_frame = ttk.Frame(self.root, padding=(8, 2))
        params_frame.pack(fill=tk.X)

        ttk.Label(params_frame, text="Temperature:").pack(side=tk.LEFT)
        self.temp_var = tk.DoubleVar(value=1.0)
        ttk.Spinbox(
            params_frame, from_=0.1, to=2.0, increment=0.1,
            textvariable=self.temp_var, width=5,
        ).pack(side=tk.LEFT, padx=(2, 12))

        ttk.Label(params_frame, text="Top-p:").pack(side=tk.LEFT)
        self.top_p_var = tk.DoubleVar(value=0.95)
        ttk.Spinbox(
            params_frame, from_=0.1, to=1.0, increment=0.05,
            textvariable=self.top_p_var, width=5,
        ).pack(side=tk.LEFT, padx=(2, 12))

        ttk.Label(params_frame, text="Top-k:").pack(side=tk.LEFT)
        self.top_k_var = tk.IntVar(value=64)
        ttk.Spinbox(
            params_frame, from_=1, to=200, increment=1,
            textvariable=self.top_k_var, width=5,
        ).pack(side=tk.LEFT, padx=(2, 12))

        ttk.Label(params_frame, text="Max tokens:").pack(side=tk.LEFT)
        self.max_tokens_var = tk.IntVar(value=2048)
        ttk.Spinbox(
            params_frame, from_=64, to=8192, increment=64,
            textvariable=self.max_tokens_var, width=6,
        ).pack(side=tk.LEFT, padx=(2, 0))

        # --- Status bar (with progress) — pack BOTTOM first ---
        status_frame = ttk.Frame(self.root, padding=(0, 0))
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self.status_var = tk.StringVar(value="Loading model...")
        ttk.Label(
            status_frame, textvariable=self.status_var, anchor=tk.W, padding=(8, 3),
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            status_frame, variable=self.progress_var, maximum=100, length=200,
        )
        self.progress_bar.pack(side=tk.LEFT, padx=(0, 8))

        self.elapsed_var = tk.StringVar(value="")
        ttk.Label(
            status_frame, textvariable=self.elapsed_var, anchor=tk.E, padding=(0, 3, 8, 3),
        ).pack(side=tk.RIGHT)

        # --- Stats bar — pack BOTTOM second ---
        self.stats_var = tk.StringVar(value="")
        self.stats_label = ttk.Label(
            self.root, textvariable=self.stats_var, anchor=tk.W, padding=(8, 2),
        )
        self.stats_label.pack(fill=tk.X, side=tk.BOTTOM)

        # --- Input bar — pack BOTTOM third ---
        input_frame = ttk.Frame(self.root, padding=(8, 4, 8, 6))
        input_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self.user_input = tk.Text(
            input_frame, height=3, wrap=tk.WORD, relief=tk.FLAT, borderwidth=1,
            padx=6, pady=4,
        )
        self.user_input.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        self.user_input.bind("<Return>", self._on_enter)
        self.user_input.bind("<Shift-Return>", lambda e: None)
        self.user_input.bind("<KeyRelease>", self._on_user_input_changed)
        self.user_input.bind("<<Paste>>", self._on_user_input_changed)

        btn_frame = ttk.Frame(input_frame)
        btn_frame.pack(side=tk.RIGHT, fill=tk.Y)

        self.send_btn = ttk.Button(btn_frame, text="Send", command=self._on_send)
        self.send_btn.pack(fill=tk.X, pady=(0, 3))

        self.behaviour_btn = ttk.Button(
            btn_frame, text="Update Behaviour", command=self._on_update_behaviour
        )
        self.behaviour_btn.pack(fill=tk.X, pady=(0, 3))

        self.clear_btn = ttk.Button(btn_frame, text="Clear", command=self._on_clear)
        self.clear_btn.pack(fill=tk.X)

        # --- Chat display — fills remaining space ---
        chat_frame = ttk.Frame(self.root, padding=(8, 4))
        chat_frame.pack(fill=tk.BOTH, expand=True)

        # Use a PanedWindow so the thinking panel can be resized
        self.chat_pane = ttk.PanedWindow(chat_frame, orient=tk.VERTICAL)
        self.chat_pane.pack(fill=tk.BOTH, expand=True)

        # Thinking panel (hidden by default, shown when thinking content arrives)
        self.thinking_frame = ttk.LabelFrame(chat_frame, text="💭 Thinking", padding=4)
        self.thinking_display = scrolledtext.ScrolledText(
            self.thinking_frame, wrap=tk.WORD, state=tk.NORMAL, relief=tk.FLAT,
            borderwidth=0, padx=8, pady=8, height=8,
        )
        self.thinking_display.pack(fill=tk.BOTH, expand=True)

        # Main chat panel
        self.main_chat_frame = ttk.Frame(chat_frame)
        self.chat_display = scrolledtext.ScrolledText(
            self.main_chat_frame, wrap=tk.WORD, state=tk.NORMAL, relief=tk.FLAT,
            borderwidth=0, padx=8, pady=8,
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True)
        self._make_readonly_display(self.thinking_display)
        self._make_readonly_display(self.chat_display)

        # Diagnostics panel (hidden by default, capture stays active)
        self.diagnostics_frame = ttk.LabelFrame(chat_frame, text="Diagnostics", padding=4)
        self.diagnostics_display = scrolledtext.ScrolledText(
            self.diagnostics_frame, wrap=tk.WORD, state=tk.NORMAL, relief=tk.FLAT,
            borderwidth=0, padx=8, pady=8, height=8,
        )
        self.diagnostics_display.pack(fill=tk.BOTH, expand=True)
        self._make_readonly_display(self.diagnostics_display)

        # Startup loading panel (shown until model loading completes)
        self.loading_frame = ttk.Frame(chat_frame, padding=32)
        self.loading_frame.columnconfigure(0, weight=1)
        self.loading_frame.rowconfigure(0, weight=1)

        loading_content = ttk.Frame(self.loading_frame, padding=24)
        loading_content.grid(row=0, column=0)

        ttk.Label(
            loading_content,
            text="Gemma 4 E2B-it Chat",
            font=(self.font_family.get(), 20, "bold"),
            anchor=tk.CENTER,
        ).pack(fill=tk.X, pady=(0, 10))

        ttk.Label(
            loading_content,
            text="Preparing the local model",
            font=(self.font_family.get(), 12),
            anchor=tk.CENTER,
        ).pack(fill=tk.X, pady=(0, 18))

        self.loading_status_label = ttk.Label(
            loading_content,
            textvariable=self.status_var,
            anchor=tk.CENTER,
            wraplength=520,
        )
        self.loading_status_label.pack(fill=tk.X, pady=(0, 10))

        self.loading_progress = ttk.Progressbar(
            loading_content,
            variable=self.progress_var,
            maximum=100,
            length=520,
        )
        self.loading_progress.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(
            loading_content,
            text=(
                "First launch may take a while if the model needs to download. "
                "Later launches still need to load weights into GPU memory."
            ),
            anchor=tk.CENTER,
            justify=tk.CENTER,
            wraplength=520,
        ).pack(fill=tk.X)

        # Initially only show the chat panel
        self.chat_pane.add(self.loading_frame, weight=3)
        self._loading_screen_visible = True
        self._thinking_visible = False

    def _make_readonly_display(self, widget: tk.Text):
        widget.configure(cursor="xterm", takefocus=True)
        widget.bind("<Key>", self._block_readonly_edit)
        widget.bind("<<Paste>>", lambda _event: "break")
        widget.bind("<<Cut>>", lambda _event: "break")
        widget.bind("<Control-a>", lambda _event, w=widget: self._select_all_text(w))
        widget.bind("<Control-A>", lambda _event, w=widget: self._select_all_text(w))

    def _block_readonly_edit(self, event):
        allowed_keys = {
            "Left", "Right", "Up", "Down", "Prior", "Next", "Home", "End",
            "Shift_L", "Shift_R", "Control_L", "Control_R", "Alt_L", "Alt_R",
            "Escape",
        }
        is_ctrl = bool(event.state & 0x4)
        if is_ctrl and event.keysym.lower() in {"a", "c", "insert"}:
            return None
        if event.keysym in allowed_keys:
            return None
        return "break"

    def _select_all_text(self, widget: tk.Text):
        widget.tag_add(tk.SEL, "1.0", tk.END)
        widget.mark_set(tk.INSERT, "1.0")
        widget.see(tk.INSERT)
        return "break"

    # ── Theme ───────────────────────────────────────────────────────────

    def _toggle_theme(self):
        self.is_dark = not self.is_dark
        sv_ttk.set_theme("dark" if self.is_dark else "light")
        self.theme_btn.configure(text="☀ Light" if self.is_dark else "🌙 Dark")
        self._apply_theme()

    def _apply_theme(self):
        palette = THEMES["dark" if self.is_dark else "light"]
        text_widgets = [
            self.chat_display,
            self.user_input,
            self.system_prompt,
            self.thinking_display,
            self.diagnostics_display,
        ]
        for w in text_widgets:
            w.configure(
                bg=palette["text_bg"],
                fg=palette["text_fg"],
                insertbackground=palette["insert_bg"],
                selectbackground=palette["select_bg"],
            )
        self._configure_chat_tags()

    # ── Fonts ───────────────────────────────────────────────────────────

    def _apply_fonts(self):
        family = self.font_family.get()
        size = self.font_size.get()
        base_font = (family, size)
        for w in [
            self.chat_display,
            self.user_input,
            self.system_prompt,
            self.thinking_display,
            self.diagnostics_display,
        ]:
            w.configure(font=base_font)
        self._configure_chat_tags()

    def _configure_chat_tags(self):
        palette = THEMES["dark" if self.is_dark else "light"]
        family = self.font_family.get()
        size = self.font_size.get()

        tags = {
            "user": {"foreground": palette["user"], "font": (family, size, "bold")},
            "assistant": {"foreground": palette["assistant"], "font": (family, size, "bold")},
            "thinking": {"foreground": palette["thinking"], "font": (family, size - 1, "italic")},
            "system_msg": {"foreground": palette["system_msg"], "font": (family, size - 1, "italic")},
            "bold": {"foreground": palette["bold"], "font": (family, size, "bold")},
            "italic": {"foreground": palette["italic"], "font": (family, size, "italic")},
            "h1": {"foreground": palette["heading"], "font": (family, size + 6, "bold")},
            "h2": {"foreground": palette["heading"], "font": (family, size + 4, "bold")},
            "h3": {"foreground": palette["heading"], "font": (family, size + 2, "bold")},
            "h4": {"foreground": palette["heading"], "font": (family, size + 1, "bold")},
            "h5": {"foreground": palette["heading"], "font": (family, size, "bold")},
            "h6": {"foreground": palette["heading"], "font": (family, size, "bold italic")},
            "code_inline": {
                "foreground": palette["code_fg"],
                "background": palette["code_bg"],
                "font": (family, size),
            },
            "code_block": {
                "foreground": palette["code_fg"],
                "background": palette["code_bg"],
                "font": (family, size),
                "lmargin1": 16,
                "lmargin2": 16,
                "rmargin": 16,
            },
            "blockquote": {
                "foreground": palette["blockquote"],
                "font": (family, size, "italic"),
                "lmargin1": 20,
                "lmargin2": 20,
            },
            "stdout": {"foreground": palette["text_fg"], "font": (family, size)},
            "stderr": {"foreground": "#f87171", "font": (family, size)},
            "diagnostic_meta": {
                "foreground": palette["system_msg"],
                "font": (family, size - 1, "italic"),
            },
        }
        for tag_name, opts in tags.items():
            self.chat_display.tag_configure(tag_name, **opts)
            self.thinking_display.tag_configure(tag_name, **opts)
            self.diagnostics_display.tag_configure(tag_name, **opts)

        self.user_input.tag_configure(
            "slash_command",
            foreground=palette["user"],
            font=(family, size, "bold"),
        )
        self.user_input.tag_configure(
            "slash_command_arg",
            foreground=palette["system_msg"],
            font=(family, size),
        )

    # ── Thinking panel helpers ──────────────────────────────────────────

    def _show_thinking_panel(self):
        if not self._thinking_visible:
            self.chat_pane.insert(0, self.thinking_frame, weight=1)
            self._thinking_visible = True

    def _hide_thinking_panel(self):
        if self._thinking_visible:
            self.chat_pane.remove(self.thinking_frame)
            self._thinking_visible = False

    def _toggle_diagnostics_panel(self):
        if self._diagnostics_visible:
            self.chat_pane.remove(self.diagnostics_frame)
            self._diagnostics_visible = False
            self.diagnostics_btn.configure(text="Diagnostics")
            return

        self.chat_pane.add(self.diagnostics_frame, weight=1)
        self._diagnostics_visible = True
        self.diagnostics_btn.configure(text="Hide Diagnostics")
        self.diagnostics_display.see(tk.END)

    def _hide_loading_screen(self):
        if not self._loading_screen_visible:
            return

        try:
            self.chat_pane.remove(self.loading_frame)
        except tk.TclError:
            pass
        self.chat_pane.add(self.main_chat_frame, weight=3)
        self._loading_screen_visible = False

    def _begin_thinking_block(self):
        self._show_thinking_panel()
        self.thinking_display.configure(state=tk.NORMAL)
        if self.thinking_display.get("1.0", "end-1c").strip():
            self.thinking_display.insert(tk.END, "\n\n")
        self.thinking_display.mark_set("thinking_block_start", tk.END)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.thinking_display.insert(tk.END, f"Thinking - {timestamp}\n\n", "thinking")
        self.thinking_display.mark_set("thinking_stream_start", tk.END)
        self.thinking_display.mark_gravity("thinking_stream_start", tk.LEFT)
        self.thinking_display.configure(state=tk.NORMAL)
        self._active_thinking_block = True
        self._has_thinking_history = True
        self.thinking_display.see(tk.END)

    def _discard_active_thinking_block(self):
        if not self._active_thinking_block:
            return

        self.thinking_display.configure(state=tk.NORMAL)
        try:
            self.thinking_display.delete("thinking_block_start", tk.END)
        except tk.TclError:
            pass
        self.thinking_display.configure(state=tk.NORMAL)
        self._active_thinking_block = False
        self._has_thinking_history = bool(
            self.thinking_display.get("1.0", "end-1c").strip()
        )
        if not self._has_thinking_history:
            self._hide_thinking_panel()

    # ── Conversation logging ───────────────────────────────────────────

    def _setup_logging(self):
        self.log_dir = self._resolve_log_dir()
        if not self.log_dir:
            self.status_var.set("Logging disabled: no transcript folder selected.")
            return

        self._load_saved_system_prompt()
        self._save_system_prompt()
        self._start_new_log()

    def _read_settings(self) -> dict:
        try:
            settings = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

        return settings if isinstance(settings, dict) else {}

    def _write_settings(self, settings: dict):
        try:
            SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
            SETTINGS_FILE.write_text(
                json.dumps(settings, indent=2),
                encoding="utf-8",
            )
        except OSError:
            pass

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

        return self.log_dir / SYSTEM_PROMPT_FILE_NAME

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

    # ── Diagnostics ────────────────────────────────────────────────────

    def _setup_diagnostics_capture(self):
        self._start_diagnostics_log()
        sys.stdout = TkLogStream(self, self._stdout_original, "stdout")
        sys.stderr = TkLogStream(self, self._stderr_original, "stderr")
        self._diagnostics_redirected = True
        self._capture_diagnostic(
            f"Diagnostics started: {datetime.now().isoformat(timespec='seconds')}\n",
            "diagnostic_meta",
        )
        if self._diagnostics_log_handle and self.diagnostics_log_path:
            self._capture_diagnostic(
                f"Diagnostics log: {self.diagnostics_log_path}\n",
                "diagnostic_meta",
            )
        else:
            self._capture_diagnostic(
                "Diagnostics file logging unavailable: no transcript folder selected.\n",
                "diagnostic_meta",
            )

    def _start_diagnostics_log(self):
        if not self.log_dir:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.diagnostics_log_path = self.log_dir / f"diagnostics_{timestamp}.log"
        counter = 2
        while self.diagnostics_log_path.exists():
            self.diagnostics_log_path = self.log_dir / f"diagnostics_{timestamp}_{counter}.log"
            counter += 1

        try:
            self._diagnostics_log_handle = self.diagnostics_log_path.open(
                "a", encoding="utf-8"
            )
        except OSError as exc:
            self._diagnostics_log_handle = None
            self._write_original_stderr(f"Diagnostics file logging disabled: {exc}\n")

    def _capture_diagnostic(self, text: str, tag: str):
        if not text:
            return

        self._write_diagnostics_file(text, tag)
        self._diagnostics_queue.put((text, tag))
        if self._diagnostics_flush_job is None:
            try:
                self._diagnostics_flush_job = self.root.after(50, self._flush_diagnostics)
            except tk.TclError:
                self._diagnostics_flush_job = None

    def _write_diagnostics_file(self, text: str, tag: str):
        if not self._diagnostics_log_handle:
            return

        try:
            if tag == "stderr":
                prefix = "[stderr] "
            elif tag == "diagnostic_meta":
                prefix = "[meta] "
            else:
                prefix = ""
            self._diagnostics_log_handle.write(prefix + text)
            self._diagnostics_log_handle.flush()
        except OSError as exc:
            self._close_diagnostics_log()
            self._write_original_stderr(f"Diagnostics file logging disabled: {exc}\n")

    def _flush_diagnostics(self):
        self._diagnostics_flush_job = None
        if not hasattr(self, "diagnostics_display"):
            return

        should_scroll = self._should_autoscroll(self.diagnostics_display)
        self.diagnostics_display.configure(state=tk.NORMAL)
        while True:
            try:
                text, tag = self._diagnostics_queue.get_nowait()
            except queue.Empty:
                break
            self.diagnostics_display.insert(tk.END, text, tag)
        self.diagnostics_display.configure(state=tk.NORMAL)
        if should_scroll:
            self.diagnostics_display.see(tk.END)

    def _restore_standard_streams(self):
        if not self._diagnostics_redirected:
            return

        sys.stdout = self._stdout_original
        sys.stderr = self._stderr_original
        self._diagnostics_redirected = False

    def _close_diagnostics_log(self):
        if not self._diagnostics_log_handle:
            return

        try:
            self._diagnostics_log_handle.close()
        except OSError:
            pass
        self._diagnostics_log_handle = None

    def _write_original_stderr(self, text: str):
        if not self._stderr_original:
            return

        try:
            self._stderr_original.write(text)
            self._stderr_original.flush()
        except Exception:
            pass

    def _cancel_stream_render_jobs(self):
        for attr in ("_response_render_job", "_thinking_render_job"):
            job = getattr(self, attr)
            if job:
                try:
                    self.root.after_cancel(job)
                except tk.TclError:
                    pass
                setattr(self, attr, None)

    def _should_autoscroll(self, widget: tk.Text) -> bool:
        if self._has_active_selection(widget):
            return False

        _, bottom = widget.yview()
        return bottom >= 0.995

    def _has_active_selection(self, widget: tk.Text) -> bool:
        try:
            widget.index(tk.SEL_FIRST)
            widget.index(tk.SEL_LAST)
            return True
        except tk.TclError:
            return False

    def _append_to_widget(self, widget: tk.Text, text: str, tag: str | None = None):
        should_scroll = self._should_autoscroll(widget)
        widget.configure(state=tk.NORMAL)
        if tag:
            widget.insert(tk.END, text, tag)
        else:
            widget.insert(tk.END, text)
        widget.configure(state=tk.NORMAL)
        if should_scroll:
            widget.see(tk.END)

    def _append_thinking(self, text: str, tag: str | None = None):
        self._append_to_widget(self.thinking_display, text, tag)

    def _append_thinking_markdown(self, text: str):
        should_scroll = self._should_autoscroll(self.thinking_display)
        self.thinking_display.configure(state=tk.NORMAL)
        MarkdownRenderer.render(self.thinking_display, text)
        self.thinking_display.configure(state=tk.NORMAL)
        if should_scroll:
            self.thinking_display.see(tk.END)

    # ── Chat helpers ────────────────────────────────────────────────────

    def _append_chat(self, text: str, tag: str | None = None):
        self._append_to_widget(self.chat_display, text, tag)

    def _append_markdown(self, text: str):
        should_scroll = self._should_autoscroll(self.chat_display)
        self.chat_display.configure(state=tk.NORMAL)
        MarkdownRenderer.render(self.chat_display, text)
        self.chat_display.configure(state=tk.NORMAL)
        if should_scroll:
            self.chat_display.see(tk.END)
        if self._stream_response_pending_newline:
            self._stream_response_pending_newline = False
            self._append_chat("\n\n")

    def _render_streamed_response_markdown(self):
        self._response_render_job = None
        if self._has_active_selection(self.chat_display):
            self._schedule_streamed_response_markdown()
            return

        should_scroll = self._should_autoscroll(self.chat_display)
        self.chat_display.configure(state=tk.NORMAL)
        try:
            self.chat_display.delete("stream_start", tk.END)
            MarkdownRenderer.render(self.chat_display, self._stream_response_text)
        except tk.TclError:
            self.chat_display.insert(tk.END, self._stream_response_text)
        self.chat_display.configure(state=tk.NORMAL)
        if should_scroll:
            self.chat_display.see(tk.END)

    def _schedule_streamed_response_markdown(self):
        if self._response_render_job:
            self.root.after_cancel(self._response_render_job)
        self._response_render_job = self.root.after(120, self._render_streamed_response_markdown)

    def _render_streamed_thinking_markdown(self):
        self._thinking_render_job = None
        if self._has_active_selection(self.thinking_display):
            self._schedule_streamed_thinking_markdown()
            return

        should_scroll = self._should_autoscroll(self.thinking_display)
        self.thinking_display.configure(state=tk.NORMAL)
        try:
            self.thinking_display.delete("thinking_stream_start", tk.END)
            MarkdownRenderer.render(self.thinking_display, self._stream_thinking_text)
        except tk.TclError:
            MarkdownRenderer.render(self.thinking_display, self._stream_thinking_text)
        self.thinking_display.configure(state=tk.NORMAL)
        if should_scroll:
            self.thinking_display.see(tk.END)

    def _schedule_streamed_thinking_markdown(self):
        if self._thinking_render_job:
            self.root.after_cancel(self._thinking_render_job)
        self._thinking_render_job = self.root.after(120, self._render_streamed_thinking_markdown)

    def _replace_streamed_thinking_with_markdown(self, thinking_text: str):
        if self._has_active_selection(self.thinking_display):
            self._schedule_streamed_thinking_markdown()
            return

        should_scroll = self._should_autoscroll(self.thinking_display)
        self.thinking_display.configure(state=tk.NORMAL)
        try:
            self.thinking_display.delete("thinking_stream_start", tk.END)
        except tk.TclError:
            pass
        MarkdownRenderer.render(self.thinking_display, thinking_text)
        self.thinking_display.configure(state=tk.NORMAL)
        self._active_thinking_block = False
        self._has_thinking_history = True
        if should_scroll:
            self.thinking_display.see(tk.END)

    def _reset_conversation(self):
        self.messages = []

    def _get_system_prompt(self) -> str:
        return self.system_prompt.get("1.0", tk.END).strip()

    def _set_system_prompt(self, text: str):
        self.system_prompt.delete("1.0", tk.END)
        self.system_prompt.insert("1.0", text)
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
            _thinking_text, response_text = _split_gemma_channels(raw_text)
            rewritten = self._clean_behaviour_rewrite(response_text or raw_text)
            if not rewritten:
                raise ValueError("The behaviour rewrite was empty.")

            self.root.after(0, lambda: self._finish_behaviour_rewrite(advice, rewritten))
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

    def _finish_behaviour_rewrite(self, advice: str, rewritten: str):
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

    # ── Stats loop ──────────────────────────────────────────────────────

    def _start_stats_loop(self):
        self.stats_var.set(self.stats.get())
        self.root.after(2000, self._start_stats_loop)

    # ── Elapsed time tracker ────────────────────────────────────────────

    def _start_elapsed_timer(self):
        self._load_start = time.perf_counter()
        self._tick_elapsed()

    def _tick_elapsed(self):
        if self._load_start > 0:
            elapsed = time.perf_counter() - self._load_start
            self.elapsed_var.set(f"{elapsed:.0f}s")
            self.root.after(500, self._tick_elapsed)

    def _stop_elapsed_timer(self):
        self._load_start = 0

    # ── Model loading ───────────────────────────────────────────────────

    def _load_model_async(self):
        def _load():
            try:
                self.root.after(0, self._start_elapsed_timer)

                # Phase 1: download with progress
                TkProgressBar._tk_progress_var = self.progress_var
                TkProgressBar._tk_status_var = self.status_var
                TkProgressBar._tk_root = self.root
                self.root.after(0, lambda: self.status_var.set("Downloading model..."))
                local_path = snapshot_download(MODEL_ID, tqdm_class=TkProgressBar)

                # Phase 2: load into GPU
                self.root.after(0, lambda: self.status_var.set("Loading weights into GPU..."))
                self.root.after(0, lambda: self.progress_bar.configure(mode="indeterminate"))
                self.root.after(0, lambda: self.loading_progress.configure(mode="indeterminate"))
                self.root.after(0, lambda: self.progress_bar.start(15))
                self.root.after(0, lambda: self.loading_progress.start(15))

                self.processor = AutoProcessor.from_pretrained(local_path)
                self.model = AutoModelForCausalLM.from_pretrained(
                    local_path,
                    dtype=torch.bfloat16,
                    device_map="auto",
                )

                def _done():
                    self.progress_bar.stop()
                    self.loading_progress.stop()
                    self.progress_bar.configure(mode="determinate")
                    self.loading_progress.configure(mode="determinate")
                    self.progress_var.set(100)
                    self._stop_elapsed_timer()
                    self._hide_loading_screen()
                    if self._pending_send:
                        self._pending_send = False
                        self.status_var.set("Model loaded. Sending queued message...")
                        self._start_generate()
                    else:
                        self.status_var.set("Model loaded. Ready to chat.")
                    self._append_chat(
                        "Model loaded. Type a message and press Send.\n\n", "system_msg"
                    )
                    self._append_log_entry(
                        "System",
                        "Model loaded. Type a message and press Send.",
                    )

                self.root.after(0, _done)

            except Exception as e:
                traceback.print_exc(file=sys.stderr)

                def _show_load_error(err=e):
                    self._stop_elapsed_timer()
                    self.progress_bar.stop()
                    self.loading_progress.stop()
                    self.progress_bar.configure(mode="determinate")
                    self.loading_progress.configure(mode="determinate")
                    self.status_var.set(f"Model load failed: {err}")
                    self._hide_loading_screen()
                    self._append_chat(f"[Error] Failed to load model: {err}\n\n", "system_msg")
                    self._append_log_entry("Error", f"Failed to load model: {err}")

                self.root.after(0, _show_load_error)

        threading.Thread(target=_load, daemon=True).start()

    # ── Input handling ──────────────────────────────────────────────────

    def _on_enter(self, event):
        if not event.state & 0x1:  # Shift not held
            self._on_send()
            return "break"

    def _on_update_behaviour(self):
        instruction = self.user_input.get("1.0", tk.END).strip()
        command_instruction = self._extract_behaviour_command(instruction)
        if command_instruction is not None:
            instruction = command_instruction

        if not instruction:
            self.status_var.set("Type a behaviour change first.")
            return

        self.user_input.delete("1.0", tk.END)
        self._highlight_user_input_commands()
        self._start_behaviour_rewrite(instruction)

    def _on_send(self):
        user_text = self.user_input.get("1.0", tk.END).strip()

        if self.updating_behaviour:
            self.status_var.set("Wait for the behaviour rewrite to finish.")
            return

        behaviour_instruction = self._extract_behaviour_command(user_text)
        if behaviour_instruction is not None:
            if behaviour_instruction:
                self.user_input.delete("1.0", tk.END)
                self._highlight_user_input_commands()
                self._start_behaviour_rewrite(behaviour_instruction)
            else:
                self.status_var.set("Usage: /behaviour <instruction>")
            return

        # If generating and no new text, treat as a stop request
        if self.generating and not user_text:
            self._stop_event.set()
            self.status_var.set("Stopping...")
            return

        # If generating with new text, steer: stop current, queue follow-up
        if self.generating and user_text:
            self._stop_event.set()
            self.user_input.delete("1.0", tk.END)
            self._highlight_user_input_commands()
            # The _finalise callback in _generate will see _pending_steer
            # and automatically start a new generation.
            self._pending_steer = user_text
            self.status_var.set("Steering — stopping current generation...")
            return

        if not user_text:
            return

        self.user_input.delete("1.0", tk.END)
        self._highlight_user_input_commands()
        self._append_chat("You: ", "user")
        self._append_chat(f"{user_text}\n\n")

        self.messages.append({"role": "user", "content": user_text})
        self._append_log_entry("User", user_text)

        if self.model is None:
            # Model still loading — queue message and show feedback
            self._pending_send = True
            self.send_btn.configure(state=tk.DISABLED)
            self.status_var.set("Message queued — waiting for model to finish loading...")
            return

        self._start_generate()

    def _start_generate(self):
        self.generating = True
        self._stop_event.clear()
        self.send_btn.configure(text="Stop")
        self.status_var.set("Generating...")
        self._start_elapsed_timer()
        self._append_chat("Gemma: ", "assistant")
        self._append_generation_settings_log(self.think_var.get())
        self._cancel_stream_render_jobs()
        self._stream_response_text = ""
        self._stream_thinking_text = ""
        self._stream_response_pending_newline = False
        if self.think_var.get():
            self._begin_thinking_block()
        # Place a mark right after "Gemma: " so we know exactly where
        # the streamed text starts (for the final markdown re-render).
        self.chat_display.configure(state=tk.NORMAL)
        self.chat_display.mark_set("stream_start", tk.END + "-1c")
        self.chat_display.mark_gravity("stream_start", tk.LEFT)
        self.chat_display.configure(state=tk.NORMAL)
        threading.Thread(target=self._generate, daemon=True).start()

    def _generate(self):
        try:
            full_messages = self._build_messages()
            enable_thinking = self.think_var.get()

            text = self.processor.apply_chat_template(
                full_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            inputs = self.processor(text=text, return_tensors="pt").to(self.model.device)

            streamer = TextIteratorStreamer(
                self.processor.tokenizer,
                skip_prompt=True,
                skip_special_tokens=False,
            )

            gen_kwargs = dict(
                **inputs,
                max_new_tokens=self.max_tokens_var.get(),
                temperature=self.temp_var.get(),
                top_p=self.top_p_var.get(),
                top_k=self.top_k_var.get(),
                do_sample=True,
                streamer=streamer,
                stopping_criteria=[_StopOnEvent(self._stop_event)],
            )

            # Run generate in its own thread so we can iterate the streamer
            gen_error: list[Exception] = []

            def _run_generate():
                try:
                    self.model.generate(**gen_kwargs)
                except Exception as exc:
                    gen_error.append(exc)
                    streamer.end()  # unblock the iterator

            threading.Thread(target=_run_generate, daemon=True).start()

            # Gemma emits channel-style markers in thinking mode.
            thought_markers = (
                "<|channel>thought",
                "<|channel|>thought",
                "<thought|>",
                "<|channel>thinking",
                "<|channel|>thinking",
                "<thinking|>",
                "<|channel>analysis",
                "<|channel|>analysis",
                "<analysis|>",
            )
            response_markers = (
                "<channel|>",
                "<|channel>response",
                "<|channel|>response",
                "<response|>",
                "<|channel>final",
                "<|channel|>final",
                "<final|>",
                "<|channel>answer",
                "<|channel|>answer",
                "<answer|>",
            )
            turn_markers = ("<turn|>", "<|turn>")
            special_tokens = tuple(
                sorted(
                    (*thought_markers, *response_markers, *turn_markers),
                    key=len,
                    reverse=True,
                )
            )

            thinking_chunks: list[str] = []
            response_chunks: list[str] = []
            raw_chunks: list[str] = []
            in_thinking = enable_thinking
            buffer = ""  # accumulates text to detect special tokens

            for chunk in streamer:
                if self._stop_event.is_set():
                    break

                raw_chunks.append(chunk)
                buffer += chunk

                # Process buffer: extract special tokens and route text
                while buffer:
                    # Check if buffer starts with any special token
                    matched_token = None
                    for st in special_tokens:
                        if buffer.startswith(st):
                            matched_token = st
                            break

                    if matched_token:
                        buffer = buffer[len(matched_token):]
                        if matched_token in thought_markers:
                            in_thinking = True
                        elif matched_token in response_markers:
                            in_thinking = False
                        elif matched_token in turn_markers:
                            in_thinking = enable_thinking
                        continue

                    # Check if buffer could be the start of a special token
                    might_match = any(
                        st.startswith(buffer) and len(buffer) < len(st)
                        for st in special_tokens
                    )
                    if might_match:
                        break  # wait for more data

                    # Emit one character
                    ch = buffer[0]
                    buffer = buffer[1:]
                    if in_thinking:
                        thinking_chunks.append(ch)
                        self.root.after(0, self._stream_thinking_chunk, ch)
                    else:
                        response_chunks.append(ch)
                        self.root.after(0, self._stream_chunk, ch)

            if gen_error:
                raise gen_error[0]

            if buffer and not any(st.startswith(buffer) for st in special_tokens):
                if in_thinking:
                    thinking_chunks.append(buffer)
                    self.root.after(0, self._stream_thinking_chunk, buffer)
                else:
                    response_chunks.append(buffer)
                    self.root.after(0, self._stream_chunk, buffer)

            was_stopped = self._stop_event.is_set()
            split_thinking_text, split_response_text = _split_gemma_channels("".join(raw_chunks))
            thinking_text = "".join(thinking_chunks).strip() or split_thinking_text
            response_text = "".join(response_chunks).strip()
            if not response_text and (not enable_thinking or split_thinking_text):
                response_text = split_response_text

            if response_text:
                response_text = response_text.replace("\\n", "\n")
            if thinking_text:
                thinking_text = thinking_text.replace("\\n", "\n")

            def _finalise():
                self._cancel_stream_render_jobs()
                # Replace the streamed plain text with properly rendered markdown
                self._replace_streamed_with_markdown(response_text)

                # Re-render thinking panel with markdown if we have thinking content
                if thinking_text:
                    self._replace_streamed_thinking_with_markdown(thinking_text)
                    self._append_log_entry("Thinking", thinking_text)
                else:
                    self._discard_active_thinking_block()

                if was_stopped and response_text:
                    # Save partial response with a marker
                    logged_response = response_text + " [interrupted]"
                    self.messages.append({"role": "assistant", "content": logged_response})
                    self._append_log_entry("Assistant", logged_response)
                elif response_text:
                    self.messages.append({"role": "assistant", "content": response_text})
                    self._append_log_entry("Assistant", response_text)

                self._stop_elapsed_timer()
                self.generating = False
                self.send_btn.configure(text="Send")

                # Check if there's a pending steer (follow-up typed during generation)
                steer_text = self._pending_steer
                self._pending_steer = None
                if steer_text:
                    self._append_chat("You: ", "user")
                    self._append_chat(f"{steer_text}\n\n")
                    self.messages.append({"role": "user", "content": steer_text})
                    self._append_log_entry("User", steer_text)
                    self._start_generate()
                elif was_stopped:
                    self.status_var.set("Generation stopped.")
                else:
                    self.status_var.set("Ready.")

            self.root.after(0, _finalise)

        except Exception as e:
            traceback.print_exc(file=sys.stderr)

            def _show_error(err=e):
                self._cancel_stream_render_jobs()
                if self._active_thinking_block and self._stream_thinking_text.strip():
                    self._replace_streamed_thinking_with_markdown(self._stream_thinking_text)
                self._append_chat(f"\n[Error] {err}\n\n", "system_msg")
                self._append_log_entry("Error", str(err))
                self.status_var.set("Error occurred.")
                self.send_btn.configure(text="Send")
                self.generating = False

            self.root.after(0, _show_error)

    def _stream_chunk(self, chunk: str):
        """Append a response token and refresh markdown after a short pause."""
        self._stream_response_text += chunk
        if self._has_active_selection(self.chat_display):
            self._append_to_widget(self.chat_display, chunk)
            return

        self._schedule_streamed_response_markdown()

    def _stream_thinking_chunk(self, chunk: str):
        """Append a thinking token and refresh markdown after a short pause."""
        self._stream_thinking_text += chunk
        if self._has_active_selection(self.thinking_display):
            self._append_to_widget(self.thinking_display, chunk)
            return

        self._schedule_streamed_thinking_markdown()

    def _replace_streamed_with_markdown(self, response_text):
        """Delete the raw streamed text and re-render with markdown formatting."""
        if self._has_active_selection(self.chat_display):
            self._stream_response_text = response_text
            self._schedule_streamed_response_markdown()
            self._stream_response_pending_newline = True
            return

        self.chat_display.configure(state=tk.NORMAL)
        try:
            self.chat_display.delete("stream_start", tk.END)
        except tk.TclError:
            pass
        self.chat_display.configure(state=tk.NORMAL)

        self._append_markdown(response_text)
        self._append_chat("\n\n")
        self._stream_response_pending_newline = False

    def _on_clear(self):
        self._cancel_stream_render_jobs()
        self._append_log_entry("System", "Conversation cleared.")
        self._reset_conversation()
        self.chat_display.configure(state=tk.NORMAL)
        self.chat_display.delete("1.0", tk.END)
        self.chat_display.configure(state=tk.NORMAL)
        self.thinking_display.configure(state=tk.NORMAL)
        self.thinking_display.delete("1.0", tk.END)
        self.thinking_display.configure(state=tk.NORMAL)
        self._has_thinking_history = False
        self._active_thinking_block = False
        self._hide_thinking_panel()
        self._start_new_log()
        self.status_var.set("Conversation cleared.")

    def _on_close(self):
        self._cancel_stream_render_jobs()
        if self._system_prompt_save_job:
            try:
                self.root.after_cancel(self._system_prompt_save_job)
            except tk.TclError:
                pass
            self._system_prompt_save_job = None
        self._save_system_prompt()
        if self._diagnostics_flush_job:
            try:
                self.root.after_cancel(self._diagnostics_flush_job)
            except tk.TclError:
                pass
            self._diagnostics_flush_job = None
            self._flush_diagnostics()
        self._capture_diagnostic(
            f"Diagnostics stopped: {datetime.now().isoformat(timespec='seconds')}\n",
            "diagnostic_meta",
        )
        if self._diagnostics_flush_job:
            try:
                self.root.after_cancel(self._diagnostics_flush_job)
            except tk.TclError:
                pass
            self._diagnostics_flush_job = None
        self._flush_diagnostics()
        self._restore_standard_streams()
        self._close_diagnostics_log()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = GemmaChat(root)
    root.mainloop()
