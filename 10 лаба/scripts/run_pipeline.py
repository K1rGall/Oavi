#!/usr/bin/env python3
"""Lab 10 speech processing pipeline: spectrogram, segmentation, DTW recognition."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy import fftpack, signal
from scipy.io import wavfile

EN_WORD_TO_SYMBOL = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "plus": "+",
}

RU_WORD_TO_SYMBOL = {
    "ноль": "0",
    "один": "1",
    "два": "2",
    "три": "3",
    "четыре": "4",
    "пять": "5",
    "шесть": "6",
    "семь": "7",
    "восемь": "8",
    "девять": "9",
    "плюс": "+",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run speech analysis for lab 10.")
    parser.add_argument("--alphabet-dir", type=Path, default=Path("audio/raw/alphabet"))
    parser.add_argument("--phone-wav", type=Path, default=Path("audio/raw/phone.wav"))
    parser.add_argument(
        "--expected-sequence",
        type=str,
        default=None,
        help="Expected symbol sequence. If omitted, tries audio/raw/phone_expected.txt",
    )
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--segments-dir", type=Path, default=Path("audio/segments"))
    return parser.parse_args()


def to_float_mono(data: np.ndarray) -> np.ndarray:
    if data.ndim > 1:
        data = data.mean(axis=1)
    if data.dtype == np.int16:
        return (data.astype(np.float32) / 32768.0).clip(-1.0, 1.0)
    if data.dtype == np.int32:
        return (data.astype(np.float32) / 2147483648.0).clip(-1.0, 1.0)
    if data.dtype == np.uint8:
        return ((data.astype(np.float32) - 128.0) / 128.0).clip(-1.0, 1.0)
    return data.astype(np.float32).clip(-1.0, 1.0)


def save_wav(path: Path, sample_rate: int, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(audio, -1.0, 1.0)
    wavfile.write(path, sample_rate, (clipped * 32767.0).astype(np.int16))


def load_wav(path: Path, target_sr: int) -> tuple[int, np.ndarray]:
    sr, data = wavfile.read(path)
    audio = to_float_mono(data)
    if sr != target_sr:
        g = math.gcd(sr, target_sr)
        up = target_sr // g
        down = sr // g
        audio = signal.resample_poly(audio, up=up, down=down).astype(np.float32)
        sr = target_sr
    return sr, audio


def normalize_symbol(name: str) -> str:
    lower = name.strip().lower()
    if lower in EN_WORD_TO_SYMBOL:
        return EN_WORD_TO_SYMBOL[lower]
    if lower in RU_WORD_TO_SYMBOL:
        return RU_WORD_TO_SYMBOL[lower]
    if lower in {"+", "plus", "pl", "p"}:
        return "+"
    digit_match = re.search(r"[0-9]", lower)
    if digit_match:
        return digit_match.group(0)
    return lower


def hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(
    sample_rate: int, n_fft: int, n_mels: int = 26, fmin: float = 40.0, fmax: float | None = None
) -> np.ndarray:
    fmax = fmax if fmax is not None else sample_rate / 2.0
    mel_points = np.linspace(hz_to_mel(np.array([fmin]))[0], hz_to_mel(np.array([fmax]))[0], n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left, center, right = bins[m - 1], bins[m], bins[m + 1]
        left = max(left, 0)
        center = max(center, left + 1)
        right = max(right, center + 1)
        for k in range(left, center):
            fb[m - 1, k] = (k - left) / (center - left)
        for k in range(center, right):
            fb[m - 1, k] = (right - k) / (right - center)
    return fb


def frame_signal(audio: np.ndarray, frame_len: int, hop_len: int) -> np.ndarray:
    if len(audio) < frame_len:
        audio = np.pad(audio, (0, frame_len - len(audio)))
    n_frames = 1 + (len(audio) - frame_len) // hop_len
    frames = np.stack([audio[i * hop_len : i * hop_len + frame_len] for i in range(n_frames)], axis=0)
    return frames


def compute_mfcc(audio: np.ndarray, sample_rate: int, n_mfcc: int = 13) -> np.ndarray:
    pre_emphasis = 0.97
    emphasized = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])

    frame_len = int(round(0.025 * sample_rate))
    hop_len = int(round(0.010 * sample_rate))
    n_fft = 512
    n_mels = 26

    frames = frame_signal(emphasized, frame_len=frame_len, hop_len=hop_len)
    window = np.hamming(frame_len).astype(np.float32)
    frames = frames * window

    spec = np.fft.rfft(frames, n=n_fft, axis=1)
    power_spec = (np.abs(spec) ** 2) / n_fft

    fb = mel_filterbank(sample_rate, n_fft=n_fft, n_mels=n_mels)
    mel_energy = np.maximum(power_spec @ fb.T, 1e-12)
    log_mel = np.log(mel_energy)

    mfcc = fftpack.dct(log_mel, axis=1, type=2, norm="ortho")[:, :n_mfcc]
    mfcc = mfcc - mfcc.mean(axis=0, keepdims=True)
    return mfcc.astype(np.float32)


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = a.shape[0], b.shape[0]
    dist = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    dp = np.full((na + 1, nb + 1), np.inf, dtype=np.float64)
    dp[0, 0] = 0.0
    for i in range(1, na + 1):
        for j in range(1, nb + 1):
            best_prev = min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
            dp[i, j] = dist[i - 1, j - 1] + best_prev
    return float(dp[na, nb] / (na + nb))


def fill_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    out = mask.copy()
    i = 0
    n = len(mask)
    while i < n:
        if out[i]:
            i += 1
            continue
        start = i
        while i < n and not out[i]:
            i += 1
        end = i
        gap = end - start
        if start > 0 and end < n and gap <= max_gap:
            out[start:end] = True
    return out


def find_segments(audio: np.ndarray, sample_rate: int) -> list[tuple[int, int]]:
    frame_len = int(round(0.020 * sample_rate))
    hop_len = int(round(0.010 * sample_rate))
    frames = frame_signal(audio, frame_len=frame_len, hop_len=hop_len)
    energy = np.mean(frames * frames, axis=1)
    energy_db = 10.0 * np.log10(np.maximum(energy, 1e-12))
    smooth = np.convolve(energy_db, np.ones(5) / 5.0, mode="same")

    floor = np.percentile(smooth, 20)
    ceiling = np.percentile(smooth, 95)
    threshold = floor + 0.22 * (ceiling - floor)
    active = smooth > threshold
    active = fill_gaps(active, max_gap=int(round(0.080 / 0.010)))

    min_frames = int(round(0.14 / 0.010))
    expand = int(round(0.04 * sample_rate))

    segments: list[tuple[int, int]] = []
    i = 0
    while i < len(active):
        if not active[i]:
            i += 1
            continue
        start = i
        while i < len(active) and active[i]:
            i += 1
        end = i
        if end - start < min_frames:
            continue
        s = max(0, start * hop_len - expand)
        e = min(len(audio), end * hop_len + frame_len + expand)
        segments.append((s, e))
    return segments


def edit_distance(a: Sequence[str], b: Sequence[str]) -> int:
    na, nb = len(a), len(b)
    dp = np.zeros((na + 1, nb + 1), dtype=np.int32)
    dp[:, 0] = np.arange(na + 1)
    dp[0, :] = np.arange(nb + 1)
    for i in range(1, na + 1):
        for j in range(1, nb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i, j] = min(
                dp[i - 1, j] + 1,
                dp[i, j - 1] + 1,
                dp[i - 1, j - 1] + cost,
            )
    return int(dp[na, nb])


def plot_spectrogram(audio: np.ndarray, sample_rate: int, out_path: Path) -> None:
    f, t, z = signal.stft(
        audio,
        fs=sample_rate,
        window="hann",
        nperseg=1024,
        noverlap=768,
        nfft=2048,
        boundary=None,
        padded=False,
    )
    power_db = 10.0 * np.log10(np.maximum(np.abs(z) ** 2, 1e-12))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 5), dpi=160)
    plt.pcolormesh(t, f[1:], power_db[1:], shading="auto", cmap="magma")
    plt.yscale("log")
    plt.ylim(60, sample_rate / 2)
    plt.xlabel("Time, s")
    plt.ylabel("Frequency, Hz (log scale)")
    plt.title("Phone track spectrogram (STFT, Hann window)")
    plt.colorbar(label="Power, dB")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_segments(audio: np.ndarray, sample_rate: int, segments: list[tuple[int, int]], labels: list[str], out_path: Path) -> None:
    t = np.arange(len(audio)) / sample_rate
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 4), dpi=160)
    plt.plot(t, audio, color="#1f77b4", linewidth=0.9)
    for idx, (start, end) in enumerate(segments):
        xs = start / sample_rate
        xe = end / sample_rate
        plt.axvspan(xs, xe, color="#ff7f0e", alpha=0.2)
        center = (xs + xe) / 2.0
        label = labels[idx] if idx < len(labels) else "?"
        plt.text(center, 0.88, label, transform=plt.gca().get_xaxis_transform(), ha="center", va="top")
    plt.xlabel("Time, s")
    plt.ylabel("Amplitude")
    plt.title("Segmentation of phone track")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_distance_heatmap(
    distance_rows: list[list[float]],
    symbols_sorted: list[str],
    out_path: Path,
) -> None:
    arr = np.array(distance_rows, dtype=np.float32)
    if arr.size == 0:
        arr = np.zeros((1, len(symbols_sorted)), dtype=np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5), dpi=160)
    im = plt.imshow(arr, aspect="auto", cmap="viridis")
    plt.colorbar(im, label="DTW distance")
    plt.xticks(np.arange(len(symbols_sorted)), symbols_sorted)
    plt.yticks(np.arange(arr.shape[0]), [f"Seg {i+1}" for i in range(arr.shape[0])])
    plt.xlabel("Template symbol")
    plt.ylabel("Segment")
    plt.title("Segment-to-template DTW distance matrix")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def load_expected(args: argparse.Namespace) -> str | None:
    if args.expected_sequence is not None:
        return args.expected_sequence.strip()
    candidate = args.phone_wav.parent / "phone_expected.txt"
    if candidate.exists():
        return candidate.read_text(encoding="utf-8").strip()
    return None


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.segments_dir.mkdir(parents=True, exist_ok=True)

    expected_sequence = load_expected(args)

    template_paths = sorted(args.alphabet_dir.glob("*.wav"))
    if not template_paths:
        raise FileNotFoundError(f"No templates found in {args.alphabet_dir}")

    template_features: dict[str, np.ndarray] = {}
    template_audio: dict[str, np.ndarray] = {}
    for path in template_paths:
        symbol = normalize_symbol(path.stem)
        sr, audio = load_wav(path, target_sr=args.sample_rate)
        if len(audio) < int(0.05 * sr):
            continue
        template_audio[symbol] = audio
        template_features[symbol] = compute_mfcc(audio, sample_rate=sr)

    if not template_features:
        raise RuntimeError("No valid template features were built.")

    sr, phone_audio = load_wav(args.phone_wav, target_sr=args.sample_rate)
    plot_spectrogram(phone_audio, sample_rate=sr, out_path=args.results_dir / "spectrogram_phone.png")

    segments = find_segments(phone_audio, sample_rate=sr)
    if not segments:
        raise RuntimeError("No speech segments detected in phone track.")

    symbols_sorted = sorted(template_features.keys(), key=lambda s: (s != "+", s))
    predicted: list[str] = []
    all_distances: list[list[float]] = []

    for idx, (start, end) in enumerate(segments):
        seg = phone_audio[start:end]
        seg_feat = compute_mfcc(seg, sample_rate=sr)
        distances = [dtw_distance(seg_feat, template_features[sym]) for sym in symbols_sorted]
        best_idx = int(np.argmin(distances))
        pred_symbol = symbols_sorted[best_idx]
        predicted.append(pred_symbol)
        all_distances.append([float(d) for d in distances])
        save_wav(args.segments_dir / f"segment_{idx + 1:02d}_{pred_symbol}.wav", sr, seg)

    plot_segments(
        phone_audio,
        sample_rate=sr,
        segments=segments,
        labels=predicted,
        out_path=args.results_dir / "segmentation.png",
    )
    plot_distance_heatmap(all_distances, symbols_sorted, out_path=args.results_dir / "distance_heatmap.png")

    expected = list(expected_sequence) if expected_sequence else None
    errors = edit_distance(predicted, expected) if expected is not None else None
    confidence = None
    if expected is not None:
        denom = max(len(expected), 1)
        confidence = max(0.0, 1.0 - errors / denom)

    payload = {
        "sample_rate": sr,
        "templates": symbols_sorted,
        "predicted_sequence": "".join(predicted),
        "predicted_symbols": predicted,
        "expected_sequence": "".join(expected) if expected is not None else None,
        "num_segments": len(segments),
        "segments_seconds": [[round(s / sr, 4), round(e / sr, 4)] for s, e in segments],
        "errors": errors,
        "confidence": confidence,
    }
    out_json = args.results_dir / "recognition.json"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Speech analysis completed.")
    print(f"Predicted sequence: {payload['predicted_sequence']}")
    if payload["expected_sequence"] is not None:
        print(f"Expected sequence : {payload['expected_sequence']}")
        print(f"Errors            : {payload['errors']}")
        print(f"Confidence        : {payload['confidence']:.3f}")
    print(f"Results directory : {args.results_dir}")


if __name__ == "__main__":
    main()

