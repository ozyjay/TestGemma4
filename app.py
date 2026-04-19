"""Desktop chat UI entrypoint for Gemma 4 E2B-it."""

import tkinter as tk

from gemma_chat.resources import _set_windows_app_id
from gemma_chat.ui import GemmaChat


def main():
    _set_windows_app_id()
    root = tk.Tk()
    GemmaChat(root)
    root.mainloop()


if __name__ == "__main__":
    main()
