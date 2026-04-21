"""Gemma model stream helpers."""

import re
import threading


class _StopOnEvent:
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
