from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


PHRASE = "\u0442\u044b \u043c\u043d\u0435 \u043d\u0440\u0430\u0432\u0438\u0448\u044c\u0441\u044f \u0441 \u043a\u0430\u0436\u0434\u044b\u043c \u0434\u043d\u0435\u043c"
FONT_PATH = Path("C:/Windows/Fonts/times.ttf")
BASE_FONT_SIZE = 96
EXPERIMENT_FONT_DELTA = 8
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
    base = output / "base"
    experiment = output / "experiment"
    report_assets = output / "report_assets"

    for path in (output, base, experiment, report_assets):
        path.mkdir(parents=True, exist_ok=True)

    return {
        "output": output,
        "base": base,
        "experiment": experiment,
        "report_assets": report_assets,
    }


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


def render_phrase_mono(phrase: str, font_path: Path, font_size: int, out_dir: Path, prefix: str) -> Path:
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
    gray_path = out_dir / f"{prefix}_gray.png"
    mono_path = out_dir / f"{prefix}_mono.bmp"
    crop.save(gray_path)
    mono.save(mono_path)
    return mono_path


def load_binary(path: Path, threshold: int = 128) -> tuple[np.ndarray, np.ndarray]:
    img = Image.open(path).convert("L")
    gray = np.array(img)
    fg = (gray < threshold).astype(np.uint8)
    return gray, fg


def detect_line_box(fg: np.ndarray) -> Box:
    h_profile = fg.sum(axis=1)
    rows = np.where(h_profile > 0)[0]
    if rows.size == 0:
        raise RuntimeError("Text rows were not found in the image.")
    y1 = int(rows.min())
    y2 = int(rows.max()) + 1
    return Box(0, y1, fg.shape[1], y2)


def segment_chars(fg: np.ndarray, line_box: Box) -> list[Box]:
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
    return boxes


def crop_symbol(fg: np.ndarray, box: Box) -> np.ndarray:
    return fg[box.y1 : box.y2, box.x1 : box.x2]


def extract_features(fg_symbol: np.ndarray) -> np.ndarray:
    h, w = fg_symbol.shape
    mass = float(fg_symbol.sum())
    if mass == 0:
        return np.zeros(5, dtype=float)

    ys, xs = np.where(fg_symbol > 0)
    x = (xs + 0.5) / w
    y = (ys + 0.5) / h

    cx = float(x.mean())
    cy = float(y.mean())
    mu20 = float(np.mean((x - cx) ** 2))
    mu02 = float(np.mean((y - cy) ** 2))
    mass_norm = mass / (h * w)
    return np.array([mass_norm, cx, cy, mu20, mu02], dtype=float)


def make_reference_features(alphabet: list[str], font_path: Path, font_size: int, threshold: int) -> dict[str, np.ndarray]:
    font = ImageFont.truetype(str(font_path), font_size)
    refs: dict[str, np.ndarray] = {}
    for ch in alphabet:
        canvas = Image.new("L", (font_size * 2, font_size * 2), color=255)
        draw = ImageDraw.Draw(canvas)
        bbox = draw.textbbox((0, 0), ch, font=font)
        draw.text((20 - bbox[0], 20 - bbox[1]), ch, fill=0, font=font)

        arr = np.array(canvas)
        ys, xs = np.where(arr < 250)
        crop = arr[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
        fg = (crop < threshold).astype(np.uint8)
        refs[ch] = extract_features(fg)
    return refs


def normalize_matrix(matrix: np.ndarray, mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    den = np.where((maxs - mins) == 0, 1.0, (maxs - mins))
    return (matrix - mins) / den


def classify_symbols(
    symbols: list[np.ndarray],
    references: dict[str, np.ndarray],
) -> tuple[list[list[tuple[str, float]]], np.ndarray]:
    ref_labels = list(references.keys())
    ref_matrix = np.vstack([references[ch] for ch in ref_labels])
    mins = ref_matrix.min(axis=0)
    maxs = ref_matrix.max(axis=0)
    ref_norm = normalize_matrix(ref_matrix, mins, maxs)

    hypotheses: list[list[tuple[str, float]]] = []
    score_matrix: list[list[float]] = []

    for symbol in symbols:
        f = extract_features(symbol).reshape(1, -1)
        f_norm = normalize_matrix(f, mins, maxs)[0]
        distances = np.linalg.norm(ref_norm - f_norm, axis=1)
        closeness = 1.0 / (1.0 + distances)

        ranked = sorted(
            [(ref_labels[i], float(closeness[i])) for i in range(len(ref_labels))],
            key=lambda x: x[1],
            reverse=True,
        )
        hypotheses.append(ranked)
        score_matrix.append([score for _, score in ranked])

    return hypotheses, np.array(score_matrix, dtype=float)


def format_hypotheses_lines(hypotheses: list[list[tuple[str, float]]]) -> list[str]:
    lines = []
    for i, ranked in enumerate(hypotheses, start=1):
        pairs = ", ".join(f"('{ch}', {score:.4f})" for ch, score in ranked)
        lines.append(f"{i}: [{pairs}]")
    return lines


def evaluate(hypotheses: list[list[tuple[str, float]]], truth: str) -> dict[str, object]:
    best = "".join(item[0][0] for item in hypotheses)
    errors = sum(1 for a, b in zip(best, truth) if a != b)
    total = len(truth)
    accuracy = (total - errors) / total * 100.0 if total else 0.0
    return {
        "recognized": best,
        "truth": truth,
        "errors": errors,
        "total": total,
        "accuracy_percent": round(accuracy, 2),
    }


def save_segmentation_image(gray: np.ndarray, boxes: list[Box], out_path: Path) -> None:
    rgb = np.stack([gray, gray, gray], axis=-1)
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    for i, b in enumerate(boxes, start=1):
        draw.rectangle((b.x1, b.y1, b.x2 - 1, b.y2 - 1), outline=(220, 20, 20), width=2)
        draw.text((b.x1, max(0, b.y1 - 22)), str(i), fill=(20, 20, 220))
    img.save(out_path)


def save_top1_grid(
    symbols: list[np.ndarray],
    hypotheses: list[list[tuple[str, float]]],
    truth: str,
    out_path: Path,
) -> None:
    cols = 6
    rows = int(np.ceil(len(symbols) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.1, rows * 2.2))
    axes_arr = np.atleast_1d(axes).reshape(rows, cols)

    for i in range(rows * cols):
        ax = axes_arr[i // cols, i % cols]
        if i < len(symbols):
            pred, score = hypotheses[i][0]
            ok = pred == truth[i]
            ax.imshow(symbols[i], cmap="gray_r")
            ax.set_title(
                f"#{i+1} true='{truth[i]}' pred='{pred}'\nS={score:.3f}",
                fontsize=8,
                color="green" if ok else "red",
            )
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def save_hypothesis_heatmap(
    hypotheses: list[list[tuple[str, float]]],
    out_path: Path,
    top_n: int = 6,
) -> None:
    m = len(hypotheses)
    n = min(top_n, len(hypotheses[0]) if hypotheses else 0)
    if m == 0 or n == 0:
        return

    data = np.array([[row[i][1] for i in range(n)] for row in hypotheses], dtype=float)
    labels = [f"{i + 1}:{row[0][0]}" for i, row in enumerate(hypotheses)]
    cols = [f"Top {i + 1}" for i in range(n)]

    plt.figure(figsize=(n * 1.2 + 2, max(5, m * 0.25)))
    im = plt.imshow(data, cmap="viridis", aspect="auto")
    plt.colorbar(im, label="Closeness")
    plt.xticks(range(n), cols)
    plt.yticks(range(m), labels)
    plt.xlabel("Hypothesis rank")
    plt.ylabel("Symbol index and best class")
    plt.title("Closest hypotheses (sorted by closeness)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_accuracy_compare(base_acc: float, exp_acc: float, out_path: Path) -> None:
    labels = ["Base font size", "Experiment font size"]
    values = [base_acc, exp_acc]
    colors = ["#2f7d32", "#1565c0"]

    plt.figure(figsize=(6.5, 4))
    bars = plt.bar(labels, values, color=colors)
    plt.ylim(0, 100)
    plt.ylabel("Accuracy, %")
    plt.title("Recognition accuracy comparison")
    for bar, v in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, v + 1.5, f"{v:.2f}%", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def run_case(
    phrase: str,
    font_size: int,
    references: dict[str, np.ndarray],
    out_dir: Path,
    prefix: str,
) -> dict[str, object]:
    mono_path = render_phrase_mono(phrase, FONT_PATH, font_size, out_dir, prefix=prefix)
    gray, fg = load_binary(mono_path, threshold=INK_THRESHOLD)
    line_box = detect_line_box(fg)
    boxes = segment_chars(fg, line_box)
    symbols = [crop_symbol(fg, b) for b in boxes]

    truth = phrase.replace(" ", "")
    hypotheses, _ = classify_symbols(symbols, references)
    metrics = evaluate(hypotheses, truth)

    save_segmentation_image(gray, boxes, out_dir / f"{prefix}_segmentation.png")
    save_top1_grid(symbols, hypotheses, truth, out_dir / f"{prefix}_top1_grid.png")
    save_hypothesis_heatmap(hypotheses, out_dir / f"{prefix}_hypotheses_heatmap.png")

    (out_dir / f"{prefix}_hypotheses.txt").write_text(
        "\n".join(format_hypotheses_lines(hypotheses)),
        encoding="utf-8",
    )
    payload = {
        "font_size": font_size,
        "boxes_xyxy": [b.as_list() for b in boxes],
        "truth_without_spaces": truth,
        "metrics": metrics,
        "top1": [row[0][0] for row in hypotheses],
    }
    (out_dir / f"{prefix}_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "mono_path": str(mono_path),
        "boxes": boxes,
        "hypotheses": hypotheses,
        "metrics": metrics,
        "truth": truth,
    }


def main() -> None:
    root = Path(__file__).resolve().parent
    dirs = ensure_dirs(root)

    alphabet = sorted(set(ch for ch in PHRASE.replace(" ", "")))
    references = make_reference_features(alphabet, FONT_PATH, BASE_FONT_SIZE, INK_THRESHOLD)

    base = run_case(
        phrase=PHRASE,
        font_size=BASE_FONT_SIZE,
        references=references,
        out_dir=dirs["base"],
        prefix="base",
    )
    experiment = run_case(
        phrase=PHRASE,
        font_size=BASE_FONT_SIZE + EXPERIMENT_FONT_DELTA,
        references=references,
        out_dir=dirs["experiment"],
        prefix="experiment",
    )

    save_accuracy_compare(
        float(base["metrics"]["accuracy_percent"]),
        float(experiment["metrics"]["accuracy_percent"]),
        dirs["report_assets"] / "accuracy_compare.png",
    )

    summary = {
        "phrase": PHRASE,
        "alphabet": alphabet,
        "features": ["mass_norm", "center_x", "center_y", "mu20", "mu02"],
        "distance": "euclidean_in_normalized_feature_space",
        "closeness": "1 / (1 + distance)",
        "base": base["metrics"],
        "experiment": experiment["metrics"],
        "experiment_font_size": BASE_FONT_SIZE + EXPERIMENT_FONT_DELTA,
    }
    (dirs["output"] / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Done.")
    print(f"Phrase: {PHRASE}")
    print(f"Alphabet ({len(alphabet)}): {''.join(alphabet)}")
    print(f"Base accuracy: {base['metrics']['accuracy_percent']}%")
    print(f"Experiment accuracy: {experiment['metrics']['accuracy_percent']}%")
    print(f"Output folder: {dirs['output']}")


if __name__ == "__main__":
    main()
