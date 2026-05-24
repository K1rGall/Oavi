from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable
import unicodedata

from fontTools.ttLib import TTFont
from PIL import Image, ImageChops, ImageDraw, ImageFont

# 43 letters in the same order as on:
# https://symbl.cc/ru/alphabets/glagolitic/
GLAGOLITIC_CODEPOINTS = [
    0x2C00, 0x2C01, 0x2C02, 0x2C03, 0x2C04, 0x2C05, 0x2C06, 0x2C07, 0x2C08,
    0x2C09, 0x2C0A, 0x2C0B, 0x2C0C, 0x2C0D, 0x2C0E, 0x2C0F, 0x2C10, 0x2C11,
    0x2C12, 0x2C13, 0x2C14, 0x2C15, 0x2C2B, 0x2C16, 0x2C17, 0x2C18, 0x2C19,
    0x2C1A, 0x2C1B, 0x2C1C, 0x2C1D, 0x2C1E, 0x2C1F, 0x2C20, 0x2C21, 0x2C22,
    0x2C23, 0x2C24, 0x2C26, 0x2C27, 0x2C28, 0x2C29, 0x2C2A,
]

DEFAULT_FONT_CANDIDATES = [
    r"C:\\Windows\\Fonts\\seguihis.ttf",  # Segoe UI Historic
    r"C:\\Windows\\Fonts\\NotoSansGlagolitic-Regular.ttf",
    r"C:\\Windows\\Fonts\\FreeSerif.ttf",
]


def supports_codepoints(font_path: Path, codepoints: Iterable[int]) -> bool:
    try:
        font = TTFont(str(font_path), lazy=True)
    except Exception:
        return False

    try:
        cmap = set()
        for table in font["cmap"].tables:
            cmap.update(table.cmap.keys())
    finally:
        font.close()

    return all(cp in cmap for cp in codepoints)


def find_font(font_arg: str | None, codepoints: list[int]) -> Path:
    checked: set[Path] = set()
    candidates: list[Path] = []

    if font_arg:
        candidates.append(Path(font_arg))

    candidates.extend(Path(p) for p in DEFAULT_FONT_CANDIDATES)

    for candidate in candidates:
        if candidate in checked:
            continue
        checked.add(candidate)
        if candidate.exists() and supports_codepoints(candidate, codepoints):
            return candidate

    windows_fonts = Path(r"C:\\Windows\\Fonts")
    if windows_fonts.exists():
        for ext in ("*.ttf", "*.otf"):
            for candidate in sorted(windows_fonts.glob(ext)):
                if candidate in checked:
                    continue
                checked.add(candidate)
                if supports_codepoints(candidate, codepoints):
                    return candidate

    raise RuntimeError(
        "Не найден шрифт с поддержкой всех символов глаголицы. "
        "Укажите путь через --font."
    )


def render_cropped_symbol(symbol: str, font: ImageFont.FreeTypeFont, padding: int = 0) -> Image.Image:
    canvas_size = max(256, font.size * 4)
    canvas = Image.new("L", (canvas_size, canvas_size), 255)
    draw = ImageDraw.Draw(canvas)

    # Draw in the center, then crop by real ink bounding box.
    draw.text((canvas_size // 2, canvas_size // 2), symbol, fill=0, font=font, anchor="mm")
    bbox = ImageChops.invert(canvas).getbbox()
    if bbox is None:
        raise RuntimeError("Пустой рендер символа, проверьте выбранный шрифт.")

    cropped = canvas.crop(bbox)

    if padding > 0:
        out = Image.new("L", (cropped.width + 2 * padding, cropped.height + 2 * padding), 255)
        out.paste(cropped, (padding, padding))
        return out

    return cropped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Генерация эталонных PNG-изображений символов глаголицы (1 символ = 1 файл)."
    )
    parser.add_argument("--font", help="Путь к .ttf/.otf шрифту")
    parser.add_argument("--font-size", type=int, default=52, help="Кегль шрифта (по умолчанию 52)")
    parser.add_argument("--padding", type=int, default=0, help="Дополнительная рамка в пикселях")
    parser.add_argument(
        "--out",
        default="glagolitic_reference_symbols",
        help="Папка с PNG-файлами (по умолчанию glagolitic_reference_symbols)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    font_path = find_font(args.font, GLAGOLITIC_CODEPOINTS)
    font = ImageFont.truetype(str(font_path), args.font_size)

    meta_path = out_dir / "symbols_metadata.csv"
    with meta_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "index",
                "filename",
                "char",
                "codepoint_hex",
                "unicode_name",
                "font",
                "font_size",
            ]
        )

        for index, codepoint in enumerate(GLAGOLITIC_CODEPOINTS, start=1):
            symbol = chr(codepoint)
            image = render_cropped_symbol(symbol, font, padding=args.padding)
            filename = f"{index:02d}_U+{codepoint:04X}.png"
            image.save(out_dir / filename)

            writer.writerow([
                index,
                filename,
                symbol,
                f"U+{codepoint:04X}",
                unicodedata.name(symbol, ""),
                font_path.name,
                args.font_size,
            ])

    print(f"Done: {len(GLAGOLITIC_CODEPOINTS)} symbols saved to: {out_dir.resolve()}")
    print(f"Font: {font_path}")


if __name__ == "__main__":
    main()
