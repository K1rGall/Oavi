from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


PHRASE = "ты мне нравишься с каждым днем"
FONT_PATH = Path("C:/Windows/Fonts/times.ttf")
FONT_SIZE = 96
INK_THRESHOLD = 128

MAX_INK_IN_GAP = 2
MIN_GAP_WIDTH = 2
MIN_CHAR_WIDTH = 5


@dataclass
class Box:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    def as_list(self) -> list[int]:
        return [self.x1, self.y1, self.x2, self.y2]


def ensure_dirs(root: Path) -> dict[str, Path]:
    output = root / "output"
    source_dir = output / "source"
    profiles_dir = output / "profiles"
    segmentation_dir = output / "segmentation"
    alphabet_dir = output / "alphabet_profiles"

    for path in (output, source_dir, profiles_dir, segmentation_dir, alphabet_dir):
        path.mkdir(parents=True, exist_ok=True)

    return {
        "output": output,
        "source": source_dir,
        "profiles": profiles_dir,
        "segmentation": segmentation_dir,
        "alphabet": alphabet_dir,
    }


def render_phrase_to_bmp(phrase: str, font_path: Path, font_size: int, out_dir: Path) -> Path:
    font = ImageFont.truetype(str(font_path), font_size)

    probe = Image.new("L", (8, 8), color=255)
    probe_draw = ImageDraw.Draw(probe)
    bbox = probe_draw.textbbox((0, 0), phrase, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    canvas = Image.new("L", (text_w + 80, text_h + 80), color=255)
    draw = ImageDraw.Draw(canvas)
    draw.text((40 - bbox[0], 40 - bbox[1]), phrase, fill=0, font=font)

    arr = np.array(canvas)
    ys, xs = np.where(arr < 250)
    crop = canvas.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))

    mono = crop.convert("1", dither=Image.Dither.NONE)

    gray_path = out_dir / "01_phrase_gray.png"
    mono_png_path = out_dir / "02_phrase_mono.png"
    mono_bmp_path = out_dir / "03_phrase_mono.bmp"

    crop.save(gray_path)
    mono.convert("L").save(mono_png_path)
    mono.save(mono_bmp_path)

    return mono_bmp_path


def load_binary(path: Path, threshold: int = 128) -> tuple[np.ndarray, np.ndarray]:
    img = Image.open(path).convert("L")
    arr = np.array(img)
    fg = (arr < threshold).astype(np.uint8)
    return arr, fg


def profile_image(
    values: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
    threshold: int | None = None,
) -> None:
    plt.figure(figsize=(12, 4))
    plt.plot(values, linewidth=1.5)
    if threshold is not None:
        plt.axhline(y=threshold, color="red", linestyle="--", linewidth=1, label=f"threshold = {threshold}")
        plt.legend()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def run_spans(mask: np.ndarray) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    i = 0
    n = mask.size
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        spans.append((i, j))
        i = j
    return spans


def thin_runs(mask: np.ndarray, min_width: int) -> np.ndarray:
    out = np.zeros_like(mask, dtype=bool)
    for left, right in run_spans(mask):
        if (right - left) >= min_width:
            out[left:right] = True
    return out


def detect_line_box(fg: np.ndarray) -> Box:
    h_profile = fg.sum(axis=1)
    rows = np.where(h_profile > 0)[0]
    if rows.size == 0:
        raise RuntimeError("No text rows detected in source image.")
    y1 = int(rows.min())
    y2 = int(rows.max()) + 1
    return Box(0, y1, fg.shape[1], y2)


def segment_chars(fg: np.ndarray, line_box: Box) -> tuple[np.ndarray, np.ndarray, list[Box]]:
    line = fg[line_box.y1 : line_box.y2, line_box.x1 : line_box.x2]
    v_profile = line.sum(axis=0)

    separator_candidates = v_profile <= MAX_INK_IN_GAP
    separators = thin_runs(separator_candidates, MIN_GAP_WIDTH)
    char_mask = ~separators

    boxes: list[Box] = []
    for left, right in run_spans(char_mask):
        if (right - left) < MIN_CHAR_WIDTH:
            continue
        sub = line[:, left:right]
        rows = np.where(sub.sum(axis=1) > 0)[0]
        if rows.size == 0:
            continue
        y1 = int(rows.min())
        y2 = int(rows.max()) + 1
        boxes.append(Box(left + line_box.x1, y1 + line_box.y1, right + line_box.x1, y2 + line_box.y1))

    return line, v_profile, boxes


def save_segmentation_visuals(gray: np.ndarray, boxes: list[Box], out_dir: Path) -> None:
    rgb = np.stack([gray, gray, gray], axis=-1)
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)

    for i, b in enumerate(boxes, start=1):
        draw.rectangle((b.x1, b.y1, b.x2 - 1, b.y2 - 1), outline=(220, 20, 20), width=2)
        draw.text((b.x1, max(0, b.y1 - 22)), str(i), fill=(20, 20, 220))

    img.save(out_dir / "06_segmentation_boxes.png")


def save_symbol_crops(fg: np.ndarray, boxes: list[Box], out_dir: Path) -> None:
    if not boxes:
        return

    count = len(boxes)
    cols = 6
    rows = int(np.ceil(count / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 2.0))
    axes_arr = np.atleast_1d(axes).reshape(rows, cols)

    for idx in range(rows * cols):
        ax = axes_arr[idx // cols, idx % cols]
        if idx < count:
            b = boxes[idx]
            crop = fg[b.y1 : b.y2, b.x1 : b.x2]
            ax.imshow(crop, cmap="gray_r")
            ax.set_title(f"#{idx + 1}", fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_dir / "07_symbol_crops.png", dpi=180)
    plt.close(fig)


def normalize_profile(profile: np.ndarray) -> np.ndarray:
    if profile.max() == 0:
        return np.zeros_like(profile, dtype=float)
    return profile / profile.max()


def build_alphabet_profiles(
    phrase: str,
    font_path: Path,
    font_size: int,
    threshold: int,
    out_dir: Path,
) -> dict[str, dict[str, list[float]]]:
    alphabet = sorted(set(ch for ch in phrase.lower() if ch.strip()))
    font = ImageFont.truetype(str(font_path), font_size)
    profiles: dict[str, dict[str, list[float]]] = {}

    for ch in alphabet:
        canvas = Image.new("L", (font_size * 2, font_size * 2), color=255)
        draw = ImageDraw.Draw(canvas)
        bbox = draw.textbbox((0, 0), ch, font=font)
        x = 20 - bbox[0]
        y = 20 - bbox[1]
        draw.text((x, y), ch, fill=0, font=font)

        arr = np.array(canvas)
        ys, xs = np.where(arr < 250)
        crop = arr[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
        fg = (crop < threshold).astype(np.uint8)

        h = fg.sum(axis=1).astype(float)
        v = fg.sum(axis=0).astype(float)
        h_n = normalize_profile(h)
        v_n = normalize_profile(v)

        profiles[ch] = {
            "horizontal": h_n.round(4).tolist(),
            "vertical": v_n.round(4).tolist(),
        }

        fig, axes = plt.subplots(1, 3, figsize=(10, 3))
        axes[0].imshow(fg, cmap="gray_r")
        axes[0].set_title(f"'{ch}'")
        axes[0].axis("off")

        axes[1].plot(h_n)
        axes[1].set_title("Horizontal")
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(v_n)
        axes[2].set_title("Vertical")
        axes[2].grid(True, alpha=0.3)

        for ax in axes[1:]:
            ax.set_ylim(0, 1.05)

        plt.tight_layout()
        safe_name = ch.encode("unicode_escape").decode("ascii").replace("\\", "_")
        plt.savefig(out_dir / f"profile_{safe_name}.png", dpi=180)
        plt.close(fig)

    return profiles


def decode_profile_label(stem: str) -> str:
    token = stem.replace("profile_", "")
    if token.startswith("_u"):
        try:
            return chr(int(token[2:], 16))
        except ValueError:
            return token
    return token


def save_profile_preview(alphabet_dir: Path, output_path: Path, max_items: int | None = None) -> None:
    images = sorted(alphabet_dir.glob("profile_*.png"))
    if max_items is not None:
        images = images[:max_items]
    if not images:
        return

    cols = 3
    rows = int(np.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 2.4))
    axes_arr = np.atleast_1d(axes).reshape(rows, cols)

    for idx in range(rows * cols):
        ax = axes_arr[idx // cols, idx % cols]
        if idx < len(images):
            img = Image.open(images[idx]).convert("RGB")
            ax.imshow(img)
            ax.set_title(decode_profile_label(images[idx].stem), fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)


def write_results_json(boxes: list[Box], profiles: dict[str, dict[str, list[float]]], out_path: Path) -> None:
    payload = {
        "boxes_xyxy": [b.as_list() for b in boxes],
        "symbols_detected": len(boxes),
        "alphabet_size": len(profiles),
        "alphabet_symbols": list(profiles.keys()),
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parent
    dirs = ensure_dirs(root)

    mono_bmp = render_phrase_to_bmp(PHRASE, FONT_PATH, FONT_SIZE, dirs["source"])
    gray, fg = load_binary(mono_bmp, threshold=INK_THRESHOLD)

    h_profile = fg.sum(axis=1)
    v_profile = fg.sum(axis=0)

    profile_image(
        h_profile,
        "Horizontal projection profile",
        "Row index",
        "Foreground pixels",
        dirs["profiles"] / "04_horizontal_profile.png",
    )
    profile_image(
        v_profile,
        "Vertical projection profile",
        "Column index",
        "Foreground pixels",
        dirs["profiles"] / "05_vertical_profile_full.png",
    )

    line_box = detect_line_box(fg)
    line, line_v_profile, boxes = segment_chars(fg, line_box)

    profile_image(
        line_v_profile,
        "Vertical profile of detected text line",
        "Column index",
        "Foreground pixels",
        dirs["profiles"] / "05_vertical_profile_line.png",
        threshold=MAX_INK_IN_GAP,
    )

    save_segmentation_visuals(gray, boxes, dirs["segmentation"])
    save_symbol_crops(fg, boxes, dirs["segmentation"])

    profiles = build_alphabet_profiles(
        phrase=PHRASE,
        font_path=FONT_PATH,
        font_size=FONT_SIZE,
        threshold=INK_THRESHOLD,
        out_dir=dirs["alphabet"],
    )
    save_profile_preview(dirs["alphabet"], dirs["alphabet"] / "08_alphabet_profiles_preview.png")

    write_results_json(boxes, profiles, dirs["output"] / "results.json")

    print("Done.")
    print(f"Phrase: {PHRASE}")
    print(f"Mono BMP: {mono_bmp}")
    print(f"Detected symbols: {len(boxes)}")
    print(f"Alphabet symbols: {''.join(profiles.keys())}")
    print(f"Output folder: {dirs['output']}")


if __name__ == "__main__":
    main()
