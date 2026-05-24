#!/usr/bin/env python3
"""Render a text file to an image to include in markdown reports."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render text to png.")
    parser.add_argument("input_file", type=Path)
    parser.add_argument("output_image", type=Path)
    parser.add_argument("--title", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    text = args.input_file.read_text(encoding="utf-8")
    lines = text.splitlines() or [""]
    max_len = max(len(line) for line in lines) if lines else 1

    width = min(20, max(8, 0.11 * max_len))
    height = min(24, max(4, 0.28 * len(lines)))

    args.output_image.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(width, height), dpi=160, facecolor="#0b1220")
    ax = fig.add_subplot(111)
    ax.set_facecolor("#0b1220")
    ax.axis("off")

    y = 0.98
    if args.title:
        ax.text(
            0.02,
            y,
            args.title,
            color="#93c5fd",
            fontsize=13,
            fontfamily="DejaVu Sans",
            va="top",
            ha="left",
            weight="bold",
        )
        y -= 0.06

    ax.text(
        0.02,
        y,
        text,
        color="#e2e8f0",
        fontsize=10.5,
        fontfamily="DejaVu Sans Mono",
        va="top",
        ha="left",
        linespacing=1.35,
    )
    plt.tight_layout(pad=0.8)
    fig.savefig(args.output_image)
    plt.close(fig)
    print(f"Saved: {args.output_image}")


if __name__ == "__main__":
    main()

