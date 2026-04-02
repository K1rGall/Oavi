from pathlib import Path

import numpy as np
from PIL import Image
from scipy.signal import convolve2d


INPUT_DIR = Path("input")
OUTPUT_DIR = Path("output")
THRESHOLD = 40  # Порог для бинаризации |G|.


KERNEL_X = np.array(
    [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1],
    ],
    dtype=np.float32,
)

KERNEL_Y = np.array(
    [
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1],
    ],
    dtype=np.float32,
)


def normalize_to_uint8(matrix: np.ndarray) -> np.ndarray:
    matrix = matrix.astype(np.float32)
    min_val = float(matrix.min())
    max_val = float(matrix.max())
    if np.isclose(max_val, min_val):
        return np.zeros_like(matrix, dtype=np.uint8)
    normalized = (matrix - min_val) * (255.0 / (max_val - min_val))
    return np.clip(normalized, 0, 255).astype(np.uint8)


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.array(image.convert("RGB"), dtype=np.uint8)


def to_grayscale(rgb: np.ndarray) -> np.ndarray:
    
    return (
        0.299 * rgb[:, :, 0].astype(np.float32)
        + 0.587 * rgb[:, :, 1].astype(np.float32)
        + 0.114 * rgb[:, :, 2].astype(np.float32)
    )


def save_gray(path: Path, data: np.ndarray) -> None:
    Image.fromarray(data.astype(np.uint8), mode="L").save(path)


def save_rgb(path: Path, data: np.ndarray) -> None:
    Image.fromarray(data.astype(np.uint8), mode="RGB").save(path)


def build_collage(parts: list[np.ndarray]) -> np.ndarray:
    
    h, w = parts[0].shape[:2]
    canvas = np.zeros((h * 2, w * 3, 3), dtype=np.uint8)
    positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    for img, (r, c) in zip(parts, positions, strict=True):
        y0, y1 = r * h, (r + 1) * h
        x0, x1 = c * w, (c + 1) * w
        if img.ndim == 2:
            img = np.repeat(img[:, :, None], 3, axis=2)
        canvas[y0:y1, x0:x1] = img
    return canvas


def process_image(image_path: Path) -> None:
    rgb = load_rgb(image_path)
    gray = to_grayscale(rgb)

    gx_raw = convolve2d(gray, KERNEL_X, mode="same", boundary="symm")
    gy_raw = convolve2d(gray, KERNEL_Y, mode="same", boundary="symm")
    g_raw = np.sqrt(gx_raw**2 + gy_raw**2)

    gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)
    gx_u8 = normalize_to_uint8(np.abs(gx_raw))
    gy_u8 = normalize_to_uint8(np.abs(gy_raw))
    g_u8 = normalize_to_uint8(g_raw)
    g_bin = np.where(g_u8 >= THRESHOLD, 255, 0).astype(np.uint8)

    stem_dir = OUTPUT_DIR / image_path.stem
    stem_dir.mkdir(parents=True, exist_ok=True)

    save_rgb(stem_dir / "1_original.png", rgb)
    save_gray(stem_dir / "2_gray.png", gray_u8)
    save_gray(stem_dir / "3_gx.png", gx_u8)
    save_gray(stem_dir / "4_gy.png", gy_u8)
    save_gray(stem_dir / "5_g.png", g_u8)
    save_gray(stem_dir / "6_binary.png", g_bin)

    collage = build_collage([rgb, gray_u8, gx_u8, gy_u8, g_u8, g_bin])
    save_rgb(stem_dir / "0_collage.png", collage)


def main() -> None:
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input directory not found: {INPUT_DIR.resolve()}")

    images = sorted(
        path
        for path in INPUT_DIR.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    )
    if not images:
        raise FileNotFoundError(f"No image files found in: {INPUT_DIR.resolve()}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for image_path in images:
        process_image(image_path)
        print(f"Processed: {image_path.name}")

    print(f"Done. Results saved to: {OUTPUT_DIR.resolve()}")
    print(f"Variant: 3 (Sobel 3x3, G = sqrt(Gx^2 + Gy^2), threshold = {THRESHOLD})")


if __name__ == "__main__":
    main()
