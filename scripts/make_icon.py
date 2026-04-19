"""Convert a transparent PNG into a Windows .ico file for PyInstaller."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


ICON_SIZES = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
PADDING_RATIO = 0.08


def prepare_icon_canvas(image: Image.Image) -> Image.Image:
    """Center visible artwork on a transparent square canvas."""
    image = image.convert("RGBA")
    alpha = image.getchannel("A")
    bbox = alpha.getbbox()
    if bbox:
        image = image.crop(bbox)

    side = max(image.size)
    padding = max(1, round(side * PADDING_RATIO))
    canvas_side = side + padding * 2
    canvas = Image.new("RGBA", (canvas_side, canvas_side), (0, 0, 0, 0))
    left = (canvas_side - image.width) // 2
    top = (canvas_side - image.height) // 2
    canvas.alpha_composite(image, (left, top))
    return canvas


def make_icon_frames(image: Image.Image) -> list[Image.Image]:
    frames: list[Image.Image] = []
    for size in ICON_SIZES:
        frame = image.resize(size, Image.Resampling.LANCZOS)
        frames.append(frame)
    return frames


def validate_transparency(path: Path):
    icon = Image.open(path)
    for size in ICON_SIZES:
        frame = icon.ico.getimage(size).convert("RGBA")
        alpha = frame.getchannel("A")
        if alpha.getextrema()[0] != 0:
            raise ValueError(f"{path} {size[0]}x{size[1]} frame has no transparent pixels")

        corners = (
            frame.getpixel((0, 0)),
            frame.getpixel((frame.width - 1, 0)),
            frame.getpixel((0, frame.height - 1)),
            frame.getpixel((frame.width - 1, frame.height - 1)),
        )
        if any(pixel[3] != 0 for pixel in corners):
            raise ValueError(f"{path} {size[0]}x{size[1]} frame has opaque corners")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(args.input)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    image = prepare_icon_canvas(Image.open(args.input))
    frames = make_icon_frames(image)
    frames[-1].save(
        args.output,
        format="ICO",
        sizes=ICON_SIZES,
        append_images=frames[:-1],
        bitmap_format="png",
    )
    validate_transparency(args.output)

    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
