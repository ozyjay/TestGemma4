"""Application constants and theme palettes."""

from pathlib import Path

MODEL_ID = "google/gemma-4-E2B-it"
APP_FOLDER_NAME = ".test.gemma4"
SYSTEM_PROMPT_FILE_NAME = "system_prompt.md"
CONVERSATION_FILE_NAME = "conversation.json"
SETTINGS_DIR = Path.home() / "AppData" / "Roaming" / "TestGemma4"
SETTINGS_FILE = SETTINGS_DIR / "settings.json"

THEMES = {
    "dark": {
        "window_bg": "#111318",
        "surface": "#171a21",
        "surface_alt": "#20242d",
        "border": "#2b313d",
        "text_bg": "#1e1e1e",
        "text_fg": "#d4d4d4",
        "insert_bg": "#d4d4d4",
        "select_bg": "#264f78",
        "muted": "#9ca3af",
        "accent": "#38bdf8",
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
        "window_bg": "#f5f7fb",
        "surface": "#ffffff",
        "surface_alt": "#eef2f7",
        "border": "#d9e0ea",
        "text_bg": "#ffffff",
        "text_fg": "#1e1e1e",
        "insert_bg": "#1e1e1e",
        "select_bg": "#add6ff",
        "muted": "#64748b",
        "accent": "#0284c7",
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
