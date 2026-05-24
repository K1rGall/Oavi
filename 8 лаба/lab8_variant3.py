import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
INPUT_DIR = RESULTS_DIR / "input_images"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

IMAGE_SIZE = 384
GRAY_LEVELS = 32
NEIGHBOR_DISTANCE = 1
EPS = 1e-12


def ensure_dirs() -> None:
    for directory in (RESULTS_DIR, INPUT_DIR, FIGURES_DIR, TABLES_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def generate_text_texture(size: int = IMAGE_SIZE) -> np.ndarray:
    image = Image.new("RGB", (size, size), (248, 245, 236))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for y in range(10, size, 28):
        for x in range(8, size, 115):
            text = "OAVI LAB8 V3"
            fill = (
                int(40 + (x / size) * 150),
                int(35 + (y / size) * 120),
                int(55 + ((x + y) / (2 * size)) * 120),
            )
            draw.text((x, y), text, font=font, fill=fill)

    # Accent lines to strengthen texture directionality.
    for x in range(0, size, 16):
        shade = int(130 + 50 * math.sin(x / 20.0))
        draw.line((x, 0, x, size), fill=(shade, shade - 15, shade - 30), width=1)

    return np.asarray(image, dtype=np.uint8)


def generate_photo_like_texture(size: int = IMAGE_SIZE) -> np.ndarray:
    y = np.linspace(0.0, 1.0, size)
    x = np.linspace(0.0, 1.0, size)
    xx, yy = np.meshgrid(x, y)

    base = 0.35 + 0.45 * yy + 0.15 * np.sin(8.0 * xx) * np.cos(7.0 * yy)
    noise = np.random.default_rng(42).normal(0.0, 0.08, size=(size, size))

    r = np.clip(base + 0.20 * np.sin(11.0 * xx) + noise, 0.0, 1.0)
    g = np.clip(base + 0.15 * np.cos(9.0 * yy) + noise * 0.8, 0.0, 1.0)
    b = np.clip(base + 0.10 * np.sin(15.0 * (xx + yy)) + noise * 0.6, 0.0, 1.0)

    rgb = np.stack((r, g, b), axis=-1)
    return (rgb * 255.0).astype(np.uint8)


def generate_geometry_texture(size: int = IMAGE_SIZE) -> np.ndarray:
    image = Image.new("RGB", (size, size), (225, 235, 245))
    draw = ImageDraw.Draw(image)

    for y in range(0, size, 12):
        color = (60 + (y % 100), 90 + ((2 * y) % 120), 120 + ((3 * y) % 100))
        draw.rectangle((0, y, size, y + 5), fill=color)

    for x in range(0, size, 32):
        draw.line((x, 0, size - x // 2, size), fill=(25, 35, 45), width=2)

    for radius in range(20, size // 2, 26):
        c = 255 - radius
        draw.ellipse(
            (
                size // 2 - radius,
                size // 2 - radius,
                size // 2 + radius,
                size // 2 + radius,
            ),
            outline=(max(40, c), max(30, c - 25), max(20, c - 45)),
            width=2,
        )

    return np.asarray(image, dtype=np.uint8)


def rgb_to_hsl(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.float64) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    cmax = np.max(rgb, axis=-1)
    cmin = np.min(rgb, axis=-1)
    delta = cmax - cmin

    l = (cmax + cmin) / 2.0

    s = np.zeros_like(l)
    nonzero = delta > 0
    s[nonzero] = delta[nonzero] / (1.0 - np.abs(2.0 * l[nonzero] - 1.0) + EPS)

    h = np.zeros_like(l)
    mask_r = nonzero & (cmax == r)
    mask_g = nonzero & (cmax == g)
    mask_b = nonzero & (cmax == b)

    h[mask_r] = ((g[mask_r] - b[mask_r]) / (delta[mask_r] + EPS)) % 6.0
    h[mask_g] = ((b[mask_g] - r[mask_g]) / (delta[mask_g] + EPS)) + 2.0
    h[mask_b] = ((r[mask_b] - g[mask_b]) / (delta[mask_b] + EPS)) + 4.0

    h /= 6.0
    h = h % 1.0

    return np.stack((h, s, l), axis=-1)


def hsl_to_rgb(hsl: np.ndarray) -> np.ndarray:
    h, s, l = hsl[..., 0], hsl[..., 1], hsl[..., 2]

    c = (1.0 - np.abs(2.0 * l - 1.0)) * s
    hh = (h * 6.0) % 6.0
    x = c * (1.0 - np.abs((hh % 2.0) - 1.0))

    r1 = np.zeros_like(h)
    g1 = np.zeros_like(h)
    b1 = np.zeros_like(h)

    masks = [
        (0.0 <= hh) & (hh < 1.0),
        (1.0 <= hh) & (hh < 2.0),
        (2.0 <= hh) & (hh < 3.0),
        (3.0 <= hh) & (hh < 4.0),
        (4.0 <= hh) & (hh < 5.0),
        (5.0 <= hh) & (hh < 6.0),
    ]

    r1[masks[0]], g1[masks[0]], b1[masks[0]] = c[masks[0]], x[masks[0]], 0.0
    r1[masks[1]], g1[masks[1]], b1[masks[1]] = x[masks[1]], c[masks[1]], 0.0
    r1[masks[2]], g1[masks[2]], b1[masks[2]] = 0.0, c[masks[2]], x[masks[2]]
    r1[masks[3]], g1[masks[3]], b1[masks[3]] = 0.0, x[masks[3]], c[masks[3]]
    r1[masks[4]], g1[masks[4]], b1[masks[4]] = x[masks[4]], 0.0, c[masks[4]]
    r1[masks[5]], g1[masks[5]], b1[masks[5]] = c[masks[5]], 0.0, x[masks[5]]

    m = l - c / 2.0

    rgb = np.stack((r1 + m, g1 + m, b1 + m), axis=-1)
    rgb = np.clip(rgb, 0.0, 1.0)
    return (rgb * 255.0).round().astype(np.uint8)


def equalize_uint8_channel(channel: np.ndarray) -> np.ndarray:
    hist = np.bincount(channel.ravel(), minlength=256)
    cdf = hist.cumsum()
    nonzero = np.nonzero(hist)[0]

    if nonzero.size == 0:
        return channel.copy()

    cdf_min = cdf[nonzero[0]]
    total = channel.size

    if total == cdf_min:
        return channel.copy()

    lut = np.round((cdf - cdf_min) / (total - cdf_min) * 255.0)
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    return lut[channel]


def quantize_gray(gray: np.ndarray, levels: int = GRAY_LEVELS) -> np.ndarray:
    q = (gray.astype(np.float64) / 256.0 * levels).astype(np.int32) + 1
    q[q > levels] = levels
    return q


def compute_ngldm(gray: np.ndarray, levels: int = GRAY_LEVELS, d: int = NEIGHBOR_DISTANCE):
    q = quantize_gray(gray, levels=levels)

    if gray.shape[0] <= 2 * d or gray.shape[1] <= 2 * d:
        raise ValueError("Image is too small for selected neighborhood distance.")

    center = q[d:-d, d:-d]
    E = center.size

    neigh_sum = np.zeros_like(center, dtype=np.float64)
    for dy in range(-d, d + 1):
        for dx in range(-d, d + 1):
            if dy == 0 and dx == 0:
                continue
            neigh_sum += q[d + dy : q.shape[0] - d + dy, d + dx : q.shape[1] - d + dx]

    neighbor_count = (2 * d + 1) ** 2 - 1
    mean_neigh = neigh_sum / neighbor_count

    n_i = np.zeros(levels, dtype=np.float64)
    s_i = np.zeros(levels, dtype=np.float64)

    center_flat = center.ravel()
    diff_flat = np.abs(center - mean_neigh).ravel()

    for i in range(1, levels + 1):
        mask = center_flat == i
        n_i[i - 1] = np.count_nonzero(mask)
        if n_i[i - 1] > 0:
            s_i[i - 1] = diff_flat[mask].sum()

    p_i = n_i / E

    ngldm = np.column_stack([p_i, s_i])

    coarseness_den = float(np.sum(p_i * s_i))
    cng = 1.0 / (coarseness_den + EPS)

    present = p_i > 0
    Gp = int(np.count_nonzero(present))
    if Gp <= 1:
        con = 0.0
    else:
        levels_arr = np.arange(1, levels + 1, dtype=np.float64)
        diff_sq = (levels_arr[:, None] - levels_arr[None, :]) ** 2
        p_outer = p_i[:, None] * p_i[None, :]
        contrast_part1 = np.sum(p_outer * diff_sq) / (Gp * (Gp - 1))
        contrast_part2 = np.sum(s_i) / (E * levels * (levels - 1))
        con = float(contrast_part1 * contrast_part2)

    stats = {
        "CNG": cng,
        "CON": con,
        "E": int(E),
        "G": int(levels),
    }
    return ngldm, stats


def save_rgb(path: Path, rgb: np.ndarray) -> None:
    Image.fromarray(rgb, mode="RGB").save(path)


def save_gray(path: Path, gray: np.ndarray) -> None:
    Image.fromarray(gray, mode="L").save(path)


def plot_case_figure(
    name: str,
    orig_rgb: np.ndarray,
    gray_before: np.ndarray,
    gray_after: np.ndarray,
    ngldm_before: np.ndarray,
    ngldm_after: np.ndarray,
    stats_before: dict,
    stats_after: dict,
) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    axes[0, 0].imshow(orig_rgb)
    axes[0, 0].set_title(f"{name}: исходное RGB")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(gray_before, cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title("Полутоновое (L до)")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(gray_after, cmap="gray", vmin=0, vmax=255)
    axes[0, 2].set_title("Контрастированное (L после)")
    axes[0, 2].axis("off")

    bins = np.arange(257)
    axes[1, 0].hist(gray_before.ravel(), bins=bins, alpha=0.6, color="tab:blue", label="до")
    axes[1, 0].hist(gray_after.ravel(), bins=bins, alpha=0.55, color="tab:orange", label="после")
    axes[1, 0].set_xlim(0, 255)
    axes[1, 0].set_title("Гистограммы яркости")
    axes[1, 0].legend(loc="upper center")

    vis_before = np.log1p(ngldm_before)
    vis_after = np.log1p(ngldm_after)

    im1 = axes[1, 1].imshow(vis_before, cmap="gray", aspect="auto")
    axes[1, 1].set_title("NGLDM до (log1p)")
    axes[1, 1].set_xlabel("[p(i), s(i)]")
    axes[1, 1].set_ylabel("Уровень i")
    fig.colorbar(im1, ax=axes[1, 1], fraction=0.046, pad=0.04)

    im2 = axes[1, 2].imshow(vis_after, cmap="gray", aspect="auto")
    axes[1, 2].set_title("NGLDM после (log1p)")
    axes[1, 2].set_xlabel("[p(i), s(i)]")
    axes[1, 2].set_ylabel("Уровень i")
    fig.colorbar(im2, ax=axes[1, 2], fraction=0.046, pad=0.04)

    fig.suptitle(
        (
            f"{name} | CNG: {stats_before['CNG']:.6e} -> {stats_after['CNG']:.6e}; "
            f"CON: {stats_before['CON']:.6e} -> {stats_after['CON']:.6e}"
        ),
        fontsize=11,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = FIGURES_DIR / f"{name}_overview.png"
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return out_path


def save_features_table(rows: list[dict]) -> Path:
    csv_path = TABLES_DIR / "features_comparison.csv"
    header = "image,CNG_before,CNG_after,CON_before,CON_after,CNG_ratio_after_to_before,CON_ratio_after_to_before\n"
    lines = [header]

    for row in rows:
        lines.append(
            (
                f"{row['image']},"
                f"{row['CNG_before']:.10e},{row['CNG_after']:.10e},"
                f"{row['CON_before']:.10e},{row['CON_after']:.10e},"
                f"{row['CNG_after'] / (row['CNG_before'] + EPS):.6f},"
                f"{row['CON_after'] / (row['CON_before'] + EPS):.6f}\n"
            )
        )

    csv_path.write_text("".join(lines), encoding="utf-8")
    return csv_path


def save_features_table_image(rows: list[dict]) -> Path:
    fig, ax = plt.subplots(figsize=(12, 2 + len(rows) * 0.7))
    ax.axis("off")

    col_labels = [
        "Изображение",
        "CNG до",
        "CNG после",
        "CON до",
        "CON после",
    ]

    cell_text = []
    for row in rows:
        cell_text.append(
            [
                row["image"],
                f"{row['CNG_before']:.3e}",
                f"{row['CNG_after']:.3e}",
                f"{row['CON_before']:.3e}",
                f"{row['CON_after']:.3e}",
            ]
        )

    table = ax.table(cellText=cell_text, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    out_path = FIGURES_DIR / "features_table.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return out_path


def apply_hist_equalization_hsl(orig_rgb: np.ndarray):
    hsl = rgb_to_hsl(orig_rgb)
    l_before = np.clip(np.round(hsl[..., 2] * 255.0), 0, 255).astype(np.uint8)
    l_after = equalize_uint8_channel(l_before)

    hsl_after = hsl.copy()
    hsl_after[..., 2] = l_after.astype(np.float64) / 255.0

    contrasted_rgb = hsl_to_rgb(hsl_after)
    return l_before, l_after, contrasted_rgb


def main() -> None:
    ensure_dirs()

    image_bank = {
        "text_texture": generate_text_texture(),
        "photo_texture": generate_photo_like_texture(),
        "geometry_texture": generate_geometry_texture(),
    }

    feature_rows = []

    for name, rgb in image_bank.items():
        l_before, l_after, rgb_after = apply_hist_equalization_hsl(rgb)

        ngldm_before, stats_before = compute_ngldm(l_before, levels=GRAY_LEVELS, d=NEIGHBOR_DISTANCE)
        ngldm_after, stats_after = compute_ngldm(l_after, levels=GRAY_LEVELS, d=NEIGHBOR_DISTANCE)

        save_rgb(INPUT_DIR / f"{name}_orig_rgb.png", rgb)
        save_gray(FIGURES_DIR / f"{name}_gray_before.png", l_before)
        save_gray(FIGURES_DIR / f"{name}_gray_after.png", l_after)
        save_rgb(FIGURES_DIR / f"{name}_rgb_after.png", rgb_after)

        np.savetxt(TABLES_DIR / f"{name}_ngldm_before.csv", ngldm_before, delimiter=",", fmt="%.10f")
        np.savetxt(TABLES_DIR / f"{name}_ngldm_after.csv", ngldm_after, delimiter=",", fmt="%.10f")

        plot_case_figure(
            name=name,
            orig_rgb=rgb,
            gray_before=l_before,
            gray_after=l_after,
            ngldm_before=ngldm_before,
            ngldm_after=ngldm_after,
            stats_before=stats_before,
            stats_after=stats_after,
        )

        feature_rows.append(
            {
                "image": name,
                "CNG_before": stats_before["CNG"],
                "CNG_after": stats_after["CNG"],
                "CON_before": stats_before["CON"],
                "CON_after": stats_after["CON"],
            }
        )

    csv_path = save_features_table(feature_rows)
    table_img = save_features_table_image(feature_rows)

    print("Done.")
    print(f"Saved figures to: {FIGURES_DIR}")
    print(f"Saved tables to:  {TABLES_DIR}")
    print(f"Summary CSV:      {csv_path}")
    print(f"Summary image:    {table_img}")


if __name__ == "__main__":
    main()
