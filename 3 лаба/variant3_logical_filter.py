from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


def to_grayscale(rgb: np.ndarray) -> np.ndarray:
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return gray.astype(np.uint8)


def otsu_threshold(gray: np.ndarray) -> int:
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = gray.size
    bins = np.arange(256, dtype=np.float64)

    sum_total = (bins * hist).sum()
    weight_bg = 0.0
    sum_bg = 0.0
    max_var = -1.0
    threshold = 127

    for t in range(256):
        weight_bg += hist[t]
        if weight_bg == 0:
            continue

        weight_fg = total - weight_bg
        if weight_fg == 0:
            break

        sum_bg += t * hist[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg

        var_between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if var_between > max_var:
            max_var = var_between
            threshold = t

    return int(threshold)


def majority_filter_3x3(binary_255: np.ndarray) -> np.ndarray:
    binary01 = (binary_255 > 0).astype(np.uint8)
    p = np.pad(binary01, 1, mode="edge")

    neighborhood_sum = (
        p[:-2, :-2]
        + p[:-2, 1:-1]
        + p[:-2, 2:]
        + p[1:-1, :-2]
        + p[1:-1, 1:-1]
        + p[1:-1, 2:]
        + p[2:, :-2]
        + p[2:, 1:-1]
        + p[2:, 2:]
    )

    return np.where(neighborhood_sum >= 5, 255, 0).astype(np.uint8)


def image_files(folder: Path) -> Iterable[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return sorted((p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts), key=lambda p: p.name)


def process_file(input_path: Path, output_dir: Path) -> tuple[str, int]:
    with Image.open(input_path) as im:
        rgb = np.array(im.convert("RGB"), dtype=np.uint8)

    gray = to_grayscale(rgb)
    threshold = otsu_threshold(gray)
    binary = np.where(gray >= threshold, 255, 0).astype(np.uint8)
    filtered = majority_filter_3x3(binary)
    xor_diff = np.bitwise_xor(binary, filtered).astype(np.uint8)

    base = input_path.stem
    Image.fromarray(binary, mode="L").save(output_dir / f"{base}_mono_input.png")
    Image.fromarray(filtered, mode="L").save(output_dir / f"{base}_filtered_variant3.png")
    Image.fromarray(xor_diff, mode="L").save(output_dir / f"{base}_xor_diff.png")
    return input_path.name, threshold


def main() -> None:
    root = Path(__file__).resolve().parent
    input_dir = root / "input"
    output_dir = root / "output_variant3"
    output_dir.mkdir(parents=True, exist_ok=True)

    files = list(image_files(input_dir))
    if not files:
        raise FileNotFoundError(f"No images found in: {input_dir}")

    lines = ["Variant 3: Logical 3x3 majority filter"]
    for f in files:
        name, threshold = process_file(f, output_dir)
        line = f"{name} | threshold={threshold}"
        print(line)
        lines.append(line)

    log_path = output_dir / "processing_log.txt"
    log_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nDone. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
