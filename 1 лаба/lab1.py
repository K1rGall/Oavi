from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def load_rgb_image(path: Path) -> np.ndarray:
    if path.suffix.lower() not in {".png", ".bmp"}:
        raise ValueError("Допустимы только PNG или BMP (не JPEG).")
    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def save_rgb(array: np.ndarray, path: Path) -> None:
    Image.fromarray(array.astype(np.uint8), mode="RGB").save(path)


def save_gray(array: np.ndarray, path: Path) -> None:
    Image.fromarray(array.astype(np.uint8), mode="L").save(path)


def to_rgb_image(array: np.ndarray) -> Image.Image:
    if array.ndim == 2:
        return Image.fromarray(array.astype(np.uint8), mode="L").convert("RGB")
    return Image.fromarray(array.astype(np.uint8), mode="RGB")


def save_before_after(before: np.ndarray, after: np.ndarray, path: Path) -> None:
    left = to_rgb_image(before)
    right = to_rgb_image(after)
    gap = 10
    canvas = Image.new(
        mode="RGB",
        size=(left.width + gap + right.width, max(left.height, right.height)),
        color=(255, 255, 255),
    )
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width + gap, 0))
    canvas.save(path)


def rgb_to_hsi(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb = image.astype(np.float64) / 255.0
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    intensity = (r + g + b) / 3.0

    min_rgb = np.minimum(np.minimum(r, g), b)
    saturation = np.zeros_like(intensity)
    nonzero_intensity = intensity > 1e-12
    saturation[nonzero_intensity] = 1.0 - (
        min_rgb[nonzero_intensity] / intensity[nonzero_intensity]
    )
    saturation = np.clip(saturation, 0.0, 1.0)

    numerator = 0.5 * ((r - g) + (r - b))
    denominator = np.sqrt((r - g) ** 2 + (r - b) * (g - b))
    denominator = np.maximum(denominator, 1e-12)
    theta = np.arccos(np.clip(numerator / denominator, -1.0, 1.0))
    hue = np.where(b <= g, theta, 2.0 * np.pi - theta)
    hue = np.nan_to_num(hue, nan=0.0)

    return hue, saturation, intensity


def hsi_to_rgb(hue: np.ndarray, saturation: np.ndarray, intensity: np.ndarray) -> np.ndarray:
    h = np.mod(hue, 2.0 * np.pi)
    s = np.clip(saturation, 0.0, 1.0)
    i = np.clip(intensity, 0.0, 1.0)

    r = np.zeros_like(i)
    g = np.zeros_like(i)
    b = np.zeros_like(i)
    eps = 1e-12

    sector0 = (h >= 0.0) & (h < 2.0 * np.pi / 3.0)
    h0 = h[sector0]
    b[sector0] = i[sector0] * (1.0 - s[sector0])
    denom0 = np.cos(np.pi / 3.0 - h0)
    denom0 = np.where(np.abs(denom0) < eps, eps, denom0)
    r[sector0] = i[sector0] * (1.0 + (s[sector0] * np.cos(h0) / denom0))
    g[sector0] = 3.0 * i[sector0] - (r[sector0] + b[sector0])

    sector1 = (h >= 2.0 * np.pi / 3.0) & (h < 4.0 * np.pi / 3.0)
    h1 = h[sector1] - 2.0 * np.pi / 3.0
    r[sector1] = i[sector1] * (1.0 - s[sector1])
    denom1 = np.cos(np.pi / 3.0 - h1)
    denom1 = np.where(np.abs(denom1) < eps, eps, denom1)
    g[sector1] = i[sector1] * (1.0 + (s[sector1] * np.cos(h1) / denom1))
    b[sector1] = 3.0 * i[sector1] - (r[sector1] + g[sector1])

    sector2 = (h >= 4.0 * np.pi / 3.0) & (h < 2.0 * np.pi)
    h2 = h[sector2] - 4.0 * np.pi / 3.0
    g[sector2] = i[sector2] * (1.0 - s[sector2])
    denom2 = np.cos(np.pi / 3.0 - h2)
    denom2 = np.where(np.abs(denom2) < eps, eps, denom2)
    b[sector2] = i[sector2] * (1.0 + (s[sector2] * np.cos(h2) / denom2))
    r[sector2] = 3.0 * i[sector2] - (g[sector2] + b[sector2])

    rgb = np.stack((r, g, b), axis=2)
    rgb = np.clip(rgb, 0.0, 1.0)
    return (rgb * 255.0).round().astype(np.uint8)


def bilinear_resize(image: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    src_h, src_w, channels = image.shape
    if new_h == src_h and new_w == src_w:
        return image.copy()

    y = np.linspace(0, src_h - 1, new_h)
    x = np.linspace(0, src_w - 1, new_w)

    y0 = np.floor(y).astype(np.int32)
    x0 = np.floor(x).astype(np.int32)
    y1 = np.clip(y0 + 1, 0, src_h - 1)
    x1 = np.clip(x0 + 1, 0, src_w - 1)

    wy = (y - y0)[:, None]
    wx = (x - x0)[None, :]

    top_left = image[y0[:, None], x0[None, :]].astype(np.float64)
    top_right = image[y0[:, None], x1[None, :]].astype(np.float64)
    bottom_left = image[y1[:, None], x0[None, :]].astype(np.float64)
    bottom_right = image[y1[:, None], x1[None, :]].astype(np.float64)

    w_tl = (1.0 - wy) * (1.0 - wx)
    w_tr = (1.0 - wy) * wx
    w_bl = wy * (1.0 - wx)
    w_br = wy * wx

    out = (
        top_left * w_tl[:, :, None]
        + top_right * w_tr[:, :, None]
        + bottom_left * w_bl[:, :, None]
        + bottom_right * w_br[:, :, None]
    )
    return np.clip(out, 0, 255).round().astype(np.uint8).reshape(new_h, new_w, channels)


def decimate(image: np.ndarray, n: int) -> np.ndarray:
    if n < 1:
        raise ValueError("Коэффициент сжатия N должен быть >= 1.")
    return image[::n, ::n, :].copy()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def choose_default_input_image() -> Path | None:
    cwd = Path.cwd()
    preferred = cwd / "input_demo.png"
    if preferred.is_file():
        return preferred

    candidates = sorted(
        p for p in cwd.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".bmp"}
    )
    if candidates:
        return candidates[0]
    return None


def run_pipeline(image_path: Path, out_dir: Path, m: int, n: int) -> None:
    if m < 1 or n < 1:
        raise ValueError("Коэффициенты M и N должны быть целыми >= 1.")

    ensure_dir(out_dir)
    images_dir = out_dir / "images"
    comparisons_dir = out_dir / "comparisons"
    ensure_dir(images_dir)
    ensure_dir(comparisons_dir)

    src = load_rgb_image(image_path)
    save_rgb(src, images_dir / "00_original.png")

    r = src[:, :, 0]
    g = src[:, :, 1]
    b = src[:, :, 2]
    save_gray(r, images_dir / "01_r_component.png")
    save_gray(g, images_dir / "02_g_component.png")
    save_gray(b, images_dir / "03_b_component.png")
    save_before_after(src, r, comparisons_dir / "01_before_after_r.png")
    save_before_after(src, g, comparisons_dir / "02_before_after_g.png")
    save_before_after(src, b, comparisons_dir / "03_before_after_b.png")

    h, s, i = rgb_to_hsi(src)
    i_gray = np.clip(i * 255.0, 0, 255).round().astype(np.uint8)
    save_gray(i_gray, images_dir / "04_hsi_intensity.png")
    save_before_after(src, i_gray, comparisons_dir / "04_before_after_intensity.png")

    i_inv = 1.0 - i
    rgb_inverted_intensity = hsi_to_rgb(h, s, i_inv)
    save_rgb(rgb_inverted_intensity, images_dir / "05_inverted_intensity_rgb.png")
    save_before_after(src, rgb_inverted_intensity, comparisons_dir / "05_before_after_inverted_intensity.png")

    upscaled = bilinear_resize(src, src.shape[0] * m, src.shape[1] * m)
    save_rgb(upscaled, images_dir / "06_upscaled_M.png")
    save_before_after(src, upscaled, comparisons_dir / "06_before_after_upscaled_M.png")

    downscaled = decimate(src, n)
    save_rgb(downscaled, images_dir / "07_downscaled_N.png")
    save_before_after(src, downscaled, comparisons_dir / "07_before_after_downscaled_N.png")

    two_pass = decimate(upscaled, n)
    save_rgb(two_pass, images_dir / "08_resampled_K_two_pass.png")
    save_before_after(src, two_pass, comparisons_dir / "08_before_after_two_pass.png")

    k = m / n
    new_h = max(1, int(round(src.shape[0] * k)))
    new_w = max(1, int(round(src.shape[1] * k)))
    one_pass = bilinear_resize(src, new_h, new_w)
    save_rgb(one_pass, images_dir / "09_resampled_K_one_pass.png")
    save_before_after(src, one_pass, comparisons_dir / "09_before_after_one_pass.png")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Лабораторная №1: цветовые модели и передискретизация "
            "(без библиотечных функций ресайза)."
        )
    )
    parser.add_argument(
        "input_image",
        type=Path,
        nargs="?",
        help="Путь к исходному PNG/BMP изображению.",
    )
    parser.add_argument(
        "--m",
        type=int,
        default=3,
        help="Коэффициент растяжения M (по умолчанию: 3).",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=2,
        help="Коэффициент сжатия N (по умолчанию: 2).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("output"),
        help="Каталог для результатов (по умолчанию: ./output).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_image = args.input_image
    if input_image is None:
        input_image = choose_default_input_image()
        if input_image is None:
            raise SystemExit(
                "Не найдено входное изображение. Передайте путь явно, например:\n"
                "py lab1.py .\\input.png --m 3 --n 2 --out .\\output"
            )
        print(f"Входное изображение не указано, использую: {input_image}")

    run_pipeline(input_image, args.out, args.m, args.n)
    print(f"Готово. Результаты сохранены в: {args.out.resolve()}")


if __name__ == "__main__":
    main()
