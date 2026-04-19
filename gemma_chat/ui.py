"""Tkinter UI for the Gemma 4 desktop chat application."""

import queue
import sys
import threading
import tkinter as tk
import tkinter.font as tkfont
from datetime import datetime
from pathlib import Path
from tkinter import scrolledtext, ttk

import sv_ttk

from .behaviour import BehaviourMixin
from .config import THEMES
from .diagnostics_mixin import DiagnosticsMixin
from .persistence import PersistenceMixin
from .resources import _resource_path
from .runtime import RuntimeMixin
from .stats import StatsMonitor
from .markdown import StreamingMarkdownState
from .streaming import StreamingDisplayMixin


class GemmaChat(
    PersistenceMixin,
    DiagnosticsMixin,
    StreamingDisplayMixin,
    BehaviourMixin,
    RuntimeMixin,
):
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Gemma 4 E2B-it Chat")
        self.root.geometry("1080x780")
        self.root.minsize(780, 560)
        self._window_icon: tk.PhotoImage | None = None

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
        self._response_stream = StreamingMarkdownState(
            "stream_start",
            "response_live_tail_start",
        )
        self._thinking_stream = StreamingMarkdownState(
            "thinking_stream_start",
            "thinking_live_tail_start",
        )
        self._selection_freeze_widgets: set[str] = set()
        self._response_render_job: str | None = None
        self._thinking_render_job: str | None = None
        self._has_thinking_history = False
        self._active_thinking_block = False
        self.log_dir: Path | None = None
        self.log_path: Path | None = None
        self._system_prompt_save_job: str | None = None
        self._system_prompt_min_lines = 2
        self._system_prompt_max_lines = 8
        self.system_prompt_lines = tk.IntVar(value=self._system_prompt_min_lines)
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

        self._apply_window_icon()
        self._build_ui()
        self._apply_theme()
        self._apply_fonts()
        self._setup_logging()
        self._load_saved_conversation()
        self._setup_diagnostics_capture()
        self.system_prompt.bind("<<Modified>>", self._on_system_prompt_modified)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._start_stats_loop()
        self._load_model_async()

    # ── UI construction ─────────────────────────────────────────────────

    def _apply_window_icon(self):
        png_path = _resource_path("assets", "app-icon.png")
        ico_path = _resource_path("assets", "app-icon.ico")

        try:
            if ico_path.exists():
                self.root.iconbitmap(default=str(ico_path))
        except tk.TclError:
            pass

        try:
            if png_path.exists():
                self._window_icon = tk.PhotoImage(file=str(png_path))
                self.root.iconphoto(True, self._window_icon)
                return
        except tk.TclError:
            self._window_icon = None

        try:
            if ico_path.exists():
                self.root.iconbitmap(str(ico_path))
        except tk.TclError:
            pass

    def _build_ui(self):
        # --- Header ---
        header = ttk.Frame(self.root, padding=(14, 12), style="Header.TFrame")
        header.pack(fill=tk.X)

        brand_frame = ttk.Frame(header, style="Header.TFrame")
        brand_frame.pack(side=tk.LEFT)

        title_frame = ttk.Frame(brand_frame, style="Header.TFrame")
        title_frame.pack(side=tk.LEFT)
        ttk.Label(
            title_frame,
            text="Gemma 4 Chat",
            style="Title.TLabel",
        ).pack(anchor=tk.W)
        ttk.Label(
            title_frame,
            text="Local model workspace",
            style="Subtitle.TLabel",
        ).pack(anchor=tk.W)

        toolbar = ttk.Frame(header, style="Header.TFrame")
        toolbar.pack(side=tk.RIGHT)

        # Theme toggle
        self.theme_btn = ttk.Button(
            toolbar, text="Light", width=8, command=self._toggle_theme
        )
        self.theme_btn.pack(side=tk.LEFT, padx=(0, 8))

        # Font family
        ttk.Label(toolbar, text="Font", style="Toolbar.TLabel").pack(side=tk.LEFT)
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
        font_combo.pack(side=tk.LEFT, padx=(6, 10))
        font_combo.bind("<<ComboboxSelected>>", lambda _: self._apply_fonts())

        # Font size
        ttk.Label(toolbar, text="Size", style="Toolbar.TLabel").pack(side=tk.LEFT)
        size_spin = ttk.Spinbox(
            toolbar,
            from_=8,
            to=24,
            increment=1,
            textvariable=self.font_size,
            width=3,
            command=self._apply_fonts,
        )
        size_spin.pack(side=tk.LEFT, padx=(6, 10))

        # Thinking toggle
        self.think_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            toolbar, text="Thinking Mode", variable=self.think_var
        ).pack(side=tk.LEFT, padx=(0, 8))

        self.diagnostics_btn = ttk.Button(
            toolbar, text="Diagnostics", command=self._toggle_diagnostics_panel
        )
        self.diagnostics_btn.pack(side=tk.LEFT, padx=(0, 8))

        self.copy_latest_btn = ttk.Button(
            toolbar, text="Copy Latest Response", command=self._copy_latest_response
        )
        self.copy_latest_btn.pack(side=tk.LEFT)

        # --- Assistant behaviour ---
        sys_frame = ttk.Frame(self.root, padding=(14, 10), style="Panel.TFrame")
        sys_frame.pack(fill=tk.X, padx=12, pady=(10, 6))
        behaviour_header = ttk.Frame(sys_frame, style="Panel.TFrame")
        behaviour_header.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(
            behaviour_header,
            text="Assistant Behaviour",
            style="Section.TLabel",
        ).pack(side=tk.LEFT)
        ttk.Label(
            behaviour_header,
            text="Lines",
            style="Toolbar.TLabel",
        ).pack(side=tk.RIGHT, padx=(8, 0))
        behaviour_lines_spin = ttk.Spinbox(
            behaviour_header,
            from_=self._system_prompt_min_lines,
            to=self._system_prompt_max_lines,
            increment=1,
            textvariable=self.system_prompt_lines,
            width=3,
            command=self._apply_system_prompt_height,
        )
        behaviour_lines_spin.pack(side=tk.RIGHT)
        behaviour_lines_spin.bind(
            "<KeyRelease>",
            lambda _event: self._apply_system_prompt_height(),
        )
        behaviour_lines_spin.bind(
            "<<Increment>>",
            lambda _event: self._apply_system_prompt_height(),
        )
        behaviour_lines_spin.bind(
            "<<Decrement>>",
            lambda _event: self._apply_system_prompt_height(),
        )

        self.system_prompt = tk.Text(
            sys_frame,
            height=self.system_prompt_lines.get(),
            wrap=tk.WORD,
            relief=tk.FLAT,
            borderwidth=0,
            padx=10,
            pady=8,
        )
        self.system_prompt.insert("1.0", "You are a helpful assistant.")
        self.system_prompt.pack(fill=tk.X)
        self._apply_system_prompt_height()

        # --- Generation params (collapsible row) ---
        params_frame = ttk.Frame(self.root, padding=(14, 8), style="Params.TFrame")
        params_frame.pack(fill=tk.X, padx=12, pady=(0, 8))
        self.temp_var = tk.DoubleVar(value=1.0)
        self.top_p_var = tk.DoubleVar(value=0.95)
        self.top_k_var = tk.IntVar(value=64)
        self.max_tokens_var = tk.IntVar(value=2048)
        self._make_generation_slider(
            params_frame,
            label="Temperature",
            variable=self.temp_var,
            from_=0.1,
            to=2.0,
            resolution=0.1,
            formatter=lambda value: f"{value:.1f}",
        )
        self._make_generation_slider(
            params_frame,
            label="Top-p",
            variable=self.top_p_var,
            from_=0.1,
            to=1.0,
            resolution=0.05,
            formatter=lambda value: f"{value:.2f}",
        )
        self._make_generation_slider(
            params_frame,
            label="Top-k",
            variable=self.top_k_var,
            from_=1,
            to=200,
            resolution=1,
            formatter=lambda value: str(int(value)),
            integer=True,
        )
        self._make_generation_slider(
            params_frame,
            label="Max tokens",
            variable=self.max_tokens_var,
            from_=64,
            to=8192,
            resolution=64,
            formatter=lambda value: str(int(value)),
            integer=True,
            expand=True,
        )

        # --- Status bar (with progress) — pack BOTTOM first ---
        status_frame = ttk.Frame(self.root, padding=(12, 6), style="Status.TFrame")
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self.status_var = tk.StringVar(value="Loading model...")
        ttk.Label(
            status_frame,
            textvariable=self.status_var,
            anchor=tk.W,
            style="Status.TLabel",
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            status_frame, variable=self.progress_var, maximum=100, length=200,
        )
        self.progress_bar.pack(side=tk.LEFT, padx=(0, 8))

        self.elapsed_var = tk.StringVar(value="")
        ttk.Label(
            status_frame,
            textvariable=self.elapsed_var,
            anchor=tk.E,
            style="Status.TLabel",
        ).pack(side=tk.RIGHT)

        # --- Stats bar — pack BOTTOM second ---
        self.stats_var = tk.StringVar(value="")
        self.stats_label = ttk.Label(
            self.root,
            textvariable=self.stats_var,
            anchor=tk.W,
            padding=(12, 3),
            style="Status.TLabel",
        )
        self.stats_label.pack(fill=tk.X, side=tk.BOTTOM)

        # --- Input bar — pack BOTTOM third ---
        input_frame = ttk.Frame(self.root, padding=(12, 8, 12, 10), style="Input.TFrame")
        input_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self.user_input = tk.Text(
            input_frame,
            height=3,
            wrap=tk.WORD,
            relief=tk.FLAT,
            borderwidth=0,
            padx=10,
            pady=8,
        )
        self.user_input.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))
        self.user_input.bind("<Return>", self._on_enter)
        self.user_input.bind("<Shift-Return>", lambda e: None)
        self.user_input.bind("<KeyRelease>", self._on_user_input_changed)
        self.user_input.bind("<<Paste>>", self._on_user_input_changed)

        btn_frame = ttk.Frame(input_frame, style="Input.TFrame")
        btn_frame.pack(side=tk.RIGHT, fill=tk.Y)

        self.send_btn = ttk.Button(btn_frame, text="Send", command=self._on_send)
        self.send_btn.pack(fill=tk.X, pady=(0, 5))

        self.behaviour_btn = ttk.Button(
            btn_frame, text="Update Behaviour", command=self._on_update_behaviour
        )
        self.behaviour_btn.pack(fill=tk.X, pady=(0, 5))

        self.clear_btn = ttk.Button(btn_frame, text="Clear Conversation", command=self._on_clear)
        self.clear_btn.pack(fill=tk.X)

        # --- Chat display — fills remaining space ---
        chat_frame = ttk.Frame(self.root, padding=(12, 6), style="App.TFrame")
        chat_frame.pack(fill=tk.BOTH, expand=True)

        # Use a PanedWindow so the thinking panel can be resized
        self.chat_pane = ttk.PanedWindow(chat_frame, orient=tk.VERTICAL)
        self.chat_pane.pack(fill=tk.BOTH, expand=True)

        # Thinking panel (hidden by default, shown when thinking content arrives)
        self.thinking_frame = ttk.Frame(chat_frame, padding=(12, 10), style="Panel.TFrame")
        ttk.Label(
            self.thinking_frame,
            text="Thinking",
            style="Section.TLabel",
        ).pack(anchor=tk.W, pady=(0, 6))
        self.thinking_display = scrolledtext.ScrolledText(
            self.thinking_frame, wrap=tk.WORD, state=tk.NORMAL, relief=tk.FLAT,
            borderwidth=0, padx=10, pady=10, height=8,
        )
        self.thinking_display.pack(fill=tk.BOTH, expand=True)

        # Main chat panel
        self.main_chat_frame = ttk.Frame(chat_frame, padding=(12, 10), style="Panel.TFrame")
        ttk.Label(
            self.main_chat_frame,
            text="Conversation",
            style="Section.TLabel",
        ).pack(anchor=tk.W, pady=(0, 6))
        self.chat_display = scrolledtext.ScrolledText(
            self.main_chat_frame, wrap=tk.WORD, state=tk.NORMAL, relief=tk.FLAT,
            borderwidth=0, padx=10, pady=10,
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True)
        self._make_readonly_display(self.thinking_display)
        self._make_readonly_display(self.chat_display)

        # Diagnostics panel (hidden by default, capture stays active)
        self.diagnostics_frame = ttk.Frame(chat_frame, padding=(12, 10), style="Panel.TFrame")
        ttk.Label(
            self.diagnostics_frame,
            text="Diagnostics",
            style="Section.TLabel",
        ).pack(anchor=tk.W, pady=(0, 6))
        self.diagnostics_display = scrolledtext.ScrolledText(
            self.diagnostics_frame, wrap=tk.WORD, state=tk.NORMAL, relief=tk.FLAT,
            borderwidth=0, padx=10, pady=10, height=8,
        )
        self.diagnostics_display.pack(fill=tk.BOTH, expand=True)
        self._make_readonly_display(self.diagnostics_display)

        # Startup loading panel (shown until model loading completes)
        self.loading_frame = ttk.Frame(chat_frame, padding=32, style="Panel.TFrame")
        self.loading_frame.columnconfigure(0, weight=1)
        self.loading_frame.rowconfigure(0, weight=1)

        loading_content = ttk.Frame(self.loading_frame, padding=24, style="Panel.TFrame")
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
        widget.bind("<ButtonPress-1>", lambda _event, w=widget: self._freeze_selection_updates(w), add="+")
        widget.bind("<B1-Motion>", lambda _event, w=widget: self._freeze_selection_updates(w), add="+")
        widget.bind("<ButtonRelease-1>", lambda _event, w=widget: self._release_selection_updates(w), add="+")

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
        self._freeze_selection_updates(widget)
        widget.tag_add(tk.SEL, "1.0", tk.END)
        widget.mark_set(tk.INSERT, "1.0")
        widget.see(tk.INSERT)
        return "break"

    # ── Theme ───────────────────────────────────────────────────────────

    def _toggle_theme(self):
        self.is_dark = not self.is_dark
        sv_ttk.set_theme("dark" if self.is_dark else "light")
        self.theme_btn.configure(text="Light" if self.is_dark else "Dark")
        self._apply_theme()

    def _apply_theme(self):
        palette = THEMES["dark" if self.is_dark else "light"]
        style = ttk.Style()
        self.root.configure(bg=palette["window_bg"])
        style.configure("App.TFrame", background=palette["window_bg"])
        style.configure("Header.TFrame", background=palette["surface"])
        style.configure("Panel.TFrame", background=palette["surface"])
        style.configure("Params.TFrame", background=palette["surface_alt"])
        style.configure("Input.TFrame", background=palette["surface"])
        style.configure("Status.TFrame", background=palette["surface_alt"])
        style.configure(
            "Header.TLabel",
            background=palette["surface"],
            foreground=palette["text_fg"],
        )
        style.configure(
            "Title.TLabel",
            background=palette["surface"],
            foreground=palette["text_fg"],
            font=(self.font_family.get(), self.font_size.get() + 4, "bold"),
        )
        style.configure(
            "Subtitle.TLabel",
            background=palette["surface"],
            foreground=palette["muted"],
            font=(self.font_family.get(), max(8, self.font_size.get() - 1)),
        )
        style.configure(
            "Section.TLabel",
            background=palette["surface"],
            foreground=palette["text_fg"],
            font=(self.font_family.get(), self.font_size.get(), "bold"),
        )
        style.configure(
            "Toolbar.TLabel",
            background=palette["surface"],
            foreground=palette["muted"],
        )
        style.configure(
            "Params.TLabel",
            background=palette["surface_alt"],
            foreground=palette["muted"],
        )
        style.configure(
            "ParamValue.TLabel",
            background=palette["surface_alt"],
            foreground=palette["text_fg"],
            font=(self.font_family.get(), self.font_size.get(), "bold"),
        )
        style.configure(
            "Status.TLabel",
            background=palette["surface_alt"],
            foreground=palette["muted"],
        )
        style.configure("TPanedwindow", background=palette["window_bg"])
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
                highlightthickness=1,
                highlightbackground=palette["border"],
                highlightcolor=palette["accent"],
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
        self._apply_theme()
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

    def _make_generation_slider(
        self,
        parent: ttk.Frame,
        label: str,
        variable: tk.Variable,
        from_: float,
        to: float,
        resolution: float,
        formatter,
        integer: bool = False,
        expand: bool = False,
    ):
        frame = ttk.Frame(parent, style="Params.TFrame")
        frame.pack(side=tk.LEFT, fill=tk.X, expand=expand, padx=(0, 18 if not expand else 0))

        header = ttk.Frame(frame, style="Params.TFrame")
        header.pack(fill=tk.X)
        ttk.Label(header, text=label, style="Params.TLabel").pack(side=tk.LEFT)

        value_var = tk.StringVar(value=formatter(float(variable.get())))
        ttk.Label(
            header,
            textvariable=value_var,
            style="ParamValue.TLabel",
            width=7,
            anchor=tk.E,
        ).pack(side=tk.RIGHT)

        def on_change(raw_value):
            value = float(raw_value)
            snapped = round((value - from_) / resolution) * resolution + from_
            snapped = max(from_, min(to, snapped))
            if integer:
                snapped = int(round(snapped))
            else:
                snapped = round(snapped, 4)
            if variable.get() != snapped:
                variable.set(snapped)
            value_var.set(formatter(float(snapped)))

        scale = ttk.Scale(
            frame,
            from_=from_,
            to=to,
            variable=variable,
            command=on_change,
        )
        scale.pack(fill=tk.X, pady=(4, 0))

    def _copy_latest_response(self):
        text = self._stream_response_text.strip()
        if not text:
            for message in reversed(self.messages):
                if message.get("role") == "assistant":
                    text = str(message.get("content", "")).strip()
                    break

        if not text:
            self.status_var.set("No assistant response to copy.")
            return

        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.status_var.set("Latest assistant response copied.")

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
        self._thinking_stream.reset()
        self.thinking_display.mark_set("thinking_stream_start", "end-1c")
        self.thinking_display.mark_gravity("thinking_stream_start", tk.LEFT)
        self.thinking_display.mark_set("thinking_live_tail_start", "end-1c")
        self.thinking_display.mark_gravity("thinking_live_tail_start", tk.LEFT)
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

