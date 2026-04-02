
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

from PIL import Image


VALID_EXT = {".bmp", ".png", ".jpg", ".jpeg"}
SCRIPT_DIR = Path(__file__).resolve().parent


def iter_input_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        if path.suffix.lower() in VALID_EXT:
            yield path
        return

    for file_path in sorted(path.iterdir()):
        if file_path.is_file() and file_path.suffix.lower() in VALID_EXT:
            yield file_path


def rgb_to_grayscale_manual(rgb_image: Image.Image) -> Image.Image:
    src = rgb_image.convert("RGB")
    width, height = src.size
    src_px = src.load()

    gray_image = Image.new("L", (width, height))
    gray_px = gray_image.load()

    for y in range(height):
        for x in range(width):
            r, g, b = src_px[x, y]
            gray = int(round(0.299 * r + 0.587 * g + 0.114 * b))
            gray_px[x, y] = gray

    return gray_image


def build_integral(gray: Image.Image) -> List[List[int]]:
    width, height = gray.size
    px = gray.load()

    integral = [[0] * (width + 1) for _ in range(height + 1)]

    for y in range(1, height + 1):
        row_sum = 0
        for x in range(1, width + 1):
            row_sum += px[x - 1, y - 1]
            integral[y][x] = integral[y - 1][x] + row_sum

    return integral


def rect_sum(integral: List[List[int]], x1: int, y1: int, x2: int, y2: int) -> int:
    return (
        integral[y2 + 1][x2 + 1]
        - integral[y1][x2 + 1]
        - integral[y2 + 1][x1]
        + integral[y1][x1]
    )


def adaptive_mean_binarization_3x3(gray: Image.Image, offset: int = 0) -> Image.Image:
    width, height = gray.size
    gray_px = gray.load()
    out = Image.new("L", (width, height))
    out_px = out.load()

    integral = build_integral(gray)
    radius = 1  # 3x3

    for y in range(height):
        y1 = max(0, y - radius)
        y2 = min(height - 1, y + radius)

        for x in range(width):
            x1 = max(0, x - radius)
            x2 = min(width - 1, x + radius)

            area = (x2 - x1 + 1) * (y2 - y1 + 1)
            local_mean = rect_sum(integral, x1, y1, x2, y2) / area
            threshold = local_mean - offset

            out_px[x, y] = 255 if gray_px[x, y] > threshold else 0

    return out


def make_comparison(original: Image.Image, gray: Image.Image, binary: Image.Image) -> Image.Image:
    w, h = original.size
    canvas = Image.new("RGB", (w * 3, h), (255, 255, 255))
    canvas.paste(original.convert("RGB"), (0, 0))
    canvas.paste(gray.convert("RGB"), (w, 0))
    canvas.paste(binary.convert("RGB"), (2 * w, 0))
    return canvas


def process_file(file_path: Path, out_dir: Path, offset: int) -> None:
    image = Image.open(file_path)
    gray = rgb_to_grayscale_manual(image)
    binary = adaptive_mean_binarization_3x3(gray, offset=offset)
    compare = make_comparison(image, gray, binary)

    stem = file_path.stem
    gray_path = out_dir / f"{stem}_gray.bmp"
    binary_path = out_dir / f"{stem}_bin_v3.png"
    compare_path = out_dir / f"{stem}_compare.png"

    gray.save(gray_path, format="BMP")
    binary.save(binary_path, format="PNG")
    compare.save(compare_path, format="PNG")

    print(f"[OK] {file_path.name}")
    print(f"     gray   -> {gray_path}")
    print(f"     binary -> {binary_path}")
    print(f"     before/after -> {compare_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ЛР2, вариант 3: grayscale + адаптивная бинаризация 3x3"
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=SCRIPT_DIR / "input",
        help="Путь к изображению или папке с изображениями (по умолчанию: папка input рядом со скриптом)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=SCRIPT_DIR / "output",
        help="Папка для результатов (по умолчанию: папка output рядом со скриптом)",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Смещение порога: T = local_mean - offset (по умолчанию 0)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input
    out_dir = args.output
    offset = args.offset

    if not input_path.exists():
        raise SystemExit(
            f"Путь не найден: {input_path}. "
            "Добавьте изображения в папку ./input или передайте путь аргументом."
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    files = list(iter_input_files(input_path))
    if not files:
        raise SystemExit(
            "Нет входных изображений. Поддерживаемые форматы: "
            + ", ".join(sorted(VALID_EXT))
        )

    for file_path in files:
        process_file(file_path, out_dir, offset=offset)


if __name__ == "__main__":
    main()
