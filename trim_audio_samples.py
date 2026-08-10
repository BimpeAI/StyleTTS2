#!/usr/bin/env python3
"""Trim N-minute samples from the start of a WAV.

Usage:
  python3 trim_audio_samples.py input.wav --minutes 5
  python3 trim_audio_samples.py input.wav --minutes 5 10
  python3 trim_audio_samples.py input.wav -m 3 7 15 -o /path/to/outdir
  python3 trim_audio_samples.py input.wav -m 5 --start-sec 120
"""

from __future__ import annotations

import argparse
import wave
from pathlib import Path


def trim_wav(
    input_path: Path,
    output_path: Path,
    duration_sec: float,
    start_sec: float = 0.0,
) -> float:
    """Write a clip of `duration_sec` starting at `start_sec`. Returns clip length in seconds."""
    with wave.open(str(input_path), "rb") as src:
        nchannels = src.getnchannels()
        sampwidth = src.getsampwidth()
        framerate = src.getframerate()
        nframes = src.getnframes()
        total_sec = nframes / float(framerate)

        if start_sec < 0:
            raise ValueError(f"start_sec must be >= 0, got {start_sec}")
        if start_sec >= total_sec:
            raise ValueError(
                f"start_sec={start_sec:.2f}s is past end of file ({total_sec:.2f}s)"
            )

        available = total_sec - start_sec
        clip_sec = min(duration_sec, available)
        if clip_sec < duration_sec:
            print(
                f"Warning: only {available:.1f}s available from start={start_sec:.1f}s; "
                f"writing {clip_sec:.1f}s instead of {duration_sec:.0f}s -> {output_path.name}"
            )

        start_frame = int(start_sec * framerate)
        n_out = int(clip_sec * framerate)
        src.setpos(start_frame)
        frames = src.readframes(n_out)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as dst:
        dst.setnchannels(nchannels)
        dst.setsampwidth(sampwidth)
        dst.setframerate(framerate)
        dst.writeframes(frames)

    return clip_sec


def _label_minutes(minutes: float) -> str:
    if float(minutes).is_integer():
        return f"{int(minutes):02d}min"
    return f"{minutes:g}min".replace(".", "p")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Trim N-minute WAV sample(s) from the start of an input file"
    )
    parser.add_argument("input", type=Path, help="Input .wav file")
    parser.add_argument(
        "-m",
        "--minutes",
        type=float,
        nargs="+",
        required=True,
        metavar="MIN",
        help="Output length(s) in minutes (e.g. --minutes 5 or --minutes 5 10)",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=Path,
        default=None,
        help="Output directory (default: same folder as input)",
    )
    parser.add_argument(
        "--start-sec",
        type=float,
        default=0.0,
        help="Where to begin trimming in the source (default: 0 = start)",
    )
    args = parser.parse_args()

    minutes_list = args.minutes
    if any(m <= 0 for m in minutes_list):
        raise SystemExit("--minutes values must be > 0")

    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        raise SystemExit(f"Input not found: {input_path}")
    if input_path.suffix.lower() != ".wav":
        raise SystemExit(f"Expected a .wav file, got: {input_path.suffix}")

    outdir = (args.outdir or input_path.parent).expanduser().resolve()
    stem = input_path.stem

    for minutes in minutes_list:
        duration_sec = minutes * 60.0
        out_path = outdir / f"{stem}_{_label_minutes(minutes)}.wav"
        clip_sec = trim_wav(input_path, out_path, duration_sec, start_sec=args.start_sec)
        print(f"Wrote {out_path} ({clip_sec:.1f}s)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
