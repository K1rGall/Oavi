#!/usr/bin/env python3
"""Generate a demo speech dataset for lab 10 using ffmpeg flite."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.io import wavfile

WORD_MAP = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
    "+": "plus",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate demo wav files.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("audio/raw"),
        help="Output root directory.",
    )
    parser.add_argument(
        "--phone-sequence",
        type=str,
        default="9031574",
        help="Digits/plus sequence for the phone recording, e.g. 9031574 or +7903.",
    )
    parser.add_argument("--sample-rate", type=int, default=16000, help="Target sample rate.")
    parser.add_argument("--pause-sec", type=float, default=0.25, help="Pause between words.")
    parser.add_argument("--voice", type=str, default="slt", help="flite voice (slt, kal, rms, awb).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def normalize_audio(data: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(data))
    if peak <= 1e-9:
        return data.astype(np.float32)
    return (0.97 * data / peak).astype(np.float32)


def to_float(audio: np.ndarray) -> np.ndarray:
    if audio.dtype == np.int16:
        return (audio.astype(np.float32) / 32768.0).clip(-1.0, 1.0)
    if audio.dtype == np.int32:
        return (audio.astype(np.float32) / 2147483648.0).clip(-1.0, 1.0)
    if audio.dtype == np.uint8:
        return ((audio.astype(np.float32) - 128.0) / 128.0).clip(-1.0, 1.0)
    return audio.astype(np.float32).clip(-1.0, 1.0)


def save_wav(path: Path, sample_rate: int, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(audio, -1.0, 1.0)
    wavfile.write(path, sample_rate, (clipped * 32767.0).astype(np.int16))


def run_ffmpeg(command: list[str]) -> None:
    proc = subprocess.run(command, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "ffmpeg command failed\n"
            f"Command: {' '.join(command)}\n"
            f"stderr:\n{proc.stderr}"
        )


def synth_word(text: str, out_path: Path, sample_rate: int, voice: str, tempo: float, gain_db: float) -> None:
    filter_str = f"atempo={tempo:.3f},volume={gain_db:.2f}dB"
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"flite=text='{text}':voice={voice}",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-af",
        filter_str,
        str(out_path),
    ]
    run_ffmpeg(cmd)


def validate_sequence(sequence: str) -> list[str]:
    symbols = list(sequence.strip())
    if not symbols:
        raise ValueError("Empty phone sequence.")
    unknown = [s for s in symbols if s not in WORD_MAP]
    if unknown:
        raise ValueError(f"Unsupported symbols in phone sequence: {unknown}")
    return symbols


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    symbols = validate_sequence(args.phone_sequence)

    root = args.output_root
    alphabet_dir = root / "alphabet"
    alphabet_dir.mkdir(parents=True, exist_ok=True)

    for symbol in WORD_MAP:
        word = WORD_MAP[symbol]
        out_path = alphabet_dir / f"{symbol}.wav"
        synth_word(
            text=word,
            out_path=out_path,
            sample_rate=args.sample_rate,
            voice=args.voice,
            tempo=float(1.0 + rng.uniform(-0.04, 0.04)),
            gain_db=float(rng.uniform(-1.5, 1.5)),
        )

    phone_parts: list[np.ndarray] = []
    silence = np.zeros(int(args.pause_sec * args.sample_rate), dtype=np.float32)
    with TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for idx, symbol in enumerate(symbols):
            word = WORD_MAP[symbol]
            temp_wav = tmp_dir / f"phone_{idx}_{symbol}.wav"
            synth_word(
                text=word,
                out_path=temp_wav,
                sample_rate=args.sample_rate,
                voice=args.voice,
                tempo=float(1.0 + rng.uniform(-0.12, 0.12)),
                gain_db=float(rng.uniform(-3.0, 2.0)),
            )
            sr, sample = wavfile.read(temp_wav)
            if sr != args.sample_rate:
                raise RuntimeError(f"Unexpected sample rate: {sr} != {args.sample_rate}")
            part = to_float(sample)
            part = part + rng.normal(0.0, 0.0025, size=part.shape).astype(np.float32)
            phone_parts.append(normalize_audio(part))
            phone_parts.append(silence.copy())

    if phone_parts:
        phone_parts = phone_parts[:-1]
    phone_track = np.concatenate(phone_parts) if phone_parts else np.zeros(1, dtype=np.float32)
    phone_track = normalize_audio(phone_track)

    phone_path = root / "phone.wav"
    save_wav(phone_path, args.sample_rate, phone_track)

    expected_path = root / "phone_expected.txt"
    expected_path.write_text("".join(symbols), encoding="utf-8")

    meta = {
        "sample_rate": args.sample_rate,
        "voice": args.voice,
        "pause_sec": args.pause_sec,
        "phone_sequence": "".join(symbols),
        "alphabet_files": [str((alphabet_dir / f"{s}.wav").as_posix()) for s in WORD_MAP],
    }
    (root / "dataset_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Demo dataset created in: {root}")
    print(f"Alphabet files: {alphabet_dir}")
    print(f"Phone file: {phone_path}")
    print(f"Expected sequence: {''.join(symbols)}")


if __name__ == "__main__":
    main()

