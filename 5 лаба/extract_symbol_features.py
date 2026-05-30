from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from PIL import Image


def parse_codepoint_from_filename(filename: str) -> str:
    match = re.search(r"U\+([0-9A-Fa-f]{4,6})", filename)
    if not match:
        return ""
    return f"U+{match.group(1).upper()}"


def load_binary_mask(path: Path, threshold: int) -> np.ndarray:
    # foreground (black symbol) -> 1, background (white) -> 0
    image = Image.open(path).convert("L")
    arr = np.array(image, dtype=np.uint8)
    mask = (arr < threshold).astype(np.uint8)
    return mask


def split_quarters(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h, w = mask.shape
    mid_y = h // 2
    mid_x = w // 2
    q1 = mask[:mid_y, :mid_x]   # top-left
    q2 = mask[:mid_y, mid_x:]   # top-right
    q3 = mask[mid_y:, :mid_x]   # bottom-left
    q4 = mask[mid_y:, mid_x:]   # bottom-right
    return q1, q2, q3, q4


def compute_features(mask: np.ndarray) -> dict[str, float | int]:
    h, w = mask.shape
    mass = mask.astype(np.float64)
    total_mass = float(mass.sum())

    if total_mass <= 0:
        raise ValueError("Symbol has zero foreground mass.")

    q1, q2, q3, q4 = split_quarters(mask)
    masses = [int(q.sum()) for q in (q1, q2, q3, q4)]
    areas = [int(q.size) for q in (q1, q2, q3, q4)]
    specific = [m / a if a > 0 else 0.0 for m, a in zip(masses, areas)]

    x_coords = np.arange(w, dtype=np.float64)[None, :]
    y_coords = np.arange(h, dtype=np.float64)[:, None]

    x_c = float((mass * x_coords).sum() / total_mass)
    y_c = float((mass * y_coords).sum() / total_mass)

    x_norm = x_c / (w - 1) if w > 1 else 0.0
    y_norm = y_c / (h - 1) if h > 1 else 0.0

    # Central axial moments of inertia relative to centroid axes.
    i_x = float((((y_coords - y_c) ** 2) * mass).sum())
    i_y = float((((x_coords - x_c) ** 2) * mass).sum())

    i_x_norm = i_x / (total_mass * max(h - 1, 1) ** 2)
    i_y_norm = i_y / (total_mass * max(w - 1, 1) ** 2)

    return {
        "width": int(w),
        "height": int(h),
        "total_mass": total_mass,
        "mass_q1": masses[0],
        "mass_q2": masses[1],
        "mass_q3": masses[2],
        "mass_q4": masses[3],
        "specific_q1": specific[0],
        "specific_q2": specific[1],
        "specific_q3": specific[2],
        "specific_q4": specific[3],
        "x_c": x_c,
        "y_c": y_c,
        "x_c_norm": x_norm,
        "y_c_norm": y_norm,
        "i_x": i_x,
        "i_y": i_y,
        "i_x_norm": i_x_norm,
        "i_y_norm": i_y_norm,
    }


def build_profiles(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # X profile: sum along rows => value per x-column
    profile_x = mask.sum(axis=0).astype(int)
    # Y profile: sum along columns => value per y-row (top -> bottom)
    profile_y = mask.sum(axis=1).astype(int)
    return profile_x, profile_y


def save_profiles_plot(
    profile_x: np.ndarray,
    profile_y: np.ndarray,
    output_path: Path,
    title_suffix: str,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), dpi=150)

    xs = np.arange(profile_x.shape[0], dtype=int)
    ys = np.arange(profile_y.shape[0], dtype=int)

    ax_x, ax_y = axes

    ax_x.bar(xs, profile_x, color="#2a6f97", edgecolor="#1f3f5b", linewidth=0.4)
    ax_x.set_title(f"Profile X ({title_suffix})")
    ax_x.set_xlabel("x index (pixel)")
    ax_x.set_ylabel("black mass")
    ax_x.xaxis.set_major_locator(MaxNLocator(nbins=12, integer=True))
    ax_x.yaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
    ax_x.grid(axis="y", linestyle="--", alpha=0.35)

    ax_y.barh(ys, profile_y, color="#40916c", edgecolor="#2d6a4f", linewidth=0.4)
    ax_y.set_title(f"Profile Y ({title_suffix})")
    ax_y.set_xlabel("black mass")
    ax_y.set_ylabel("y index (top \u2192 bottom)")
    ax_y.invert_yaxis()
    ax_y.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
    ax_y.yaxis.set_major_locator(MaxNLocator(nbins=12, integer=True))
    ax_y.grid(axis="x", linestyle="--", alpha=0.35)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Lab 5 symbol features and save scalar/features profiles."
    )
    parser.add_argument(
        "--input-dir",
        default="glagolitic_reference_symbols",
        help="Directory with symbol PNG files.",
    )
    parser.add_argument(
        "--output-dir",
        default="lab5_features",
        help="Output directory for CSV and profile PNG files.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=250,
        help="Binary threshold: pixel < threshold treated as black.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    profiles_dir = output_dir / "profiles_xy"
    output_dir.mkdir(parents=True, exist_ok=True)
    profiles_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(input_dir.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No PNG files found in {input_dir.resolve()}")

    rows: list[dict[str, float | int | str]] = []
    for image_path in image_paths:
        mask = load_binary_mask(image_path, threshold=args.threshold)
        features = compute_features(mask)
        profile_x, profile_y = build_profiles(mask)

        codepoint = parse_codepoint_from_filename(image_path.name)
        symbol_id = image_path.stem

        profile_name = f"{symbol_id}_profile_xy.png"
        save_profiles_plot(
            profile_x,
            profile_y,
            profiles_dir / profile_name,
            title_suffix=symbol_id,
        )

        row: dict[str, float | int | str] = {
            "symbol_id": symbol_id,
            "filename": image_path.name,
            "codepoint": codepoint,
        }
        row.update(features)
        row["profile_file"] = f"profiles_xy/{profile_name}"
        rows.append(row)

    csv_path = output_dir / "scalar_features.csv"
    fieldnames = [
        "symbol_id",
        "filename",
        "codepoint",
        "width",
        "height",
        "total_mass",
        "mass_q1",
        "mass_q2",
        "mass_q3",
        "mass_q4",
        "specific_q1",
        "specific_q2",
        "specific_q3",
        "specific_q4",
        "x_c",
        "y_c",
        "x_c_norm",
        "y_c_norm",
        "i_x",
        "i_y",
        "i_x_norm",
        "i_y_norm",
        "profile_file",
    ]

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Done: processed {len(rows)} symbols")
    print(f"CSV: {csv_path.resolve()}")
    print(f"Profiles: {profiles_dir.resolve()}")


if __name__ == "__main__":
    main()
