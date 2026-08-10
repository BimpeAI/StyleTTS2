#!/usr/bin/env python3
"""Convert .m4a audio files to .wav.

Requires ffmpeg on PATH (pydub uses it for m4a decode):
  macOS:  brew install ffmpeg
  Ubuntu: sudo apt-get install -y ffmpeg

Examples:
  python scripts/m4a_to_wav.py recording.m4a
  python scripts/m4a_to_wav.py ./input_dir -o ./wavs --sr 24000
  python scripts/m4a_to_wav.py a.m4a b.m4a -o ./out
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pydub import AudioSegment


def convert_one(src: Path, dst: Path, sr: int | None) -> None:
    audio = AudioSegment.from_file(src, format="m4a")
    if sr is not None:
        audio = audio.set_frame_rate(sr)
    # StyleTTS2 / most TTS pipelines expect mono 16-bit PCM
    audio = audio.set_channels(1).set_sample_width(2)
    dst.parent.mkdir(parents=True, exist_ok=True)
    audio.export(dst, format="wav")


def collect_inputs(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for p in paths:
        if p.is_dir():
            files.extend(sorted(p.rglob("*.m4a")))
            files.extend(sorted(p.rglob("*.M4A")))
        elif p.is_file():
            if p.suffix.lower() != ".m4a":
                raise SystemExit(f"Not an .m4a file: {p}")
            files.append(p)
        else:
            raise SystemExit(f"Path not found: {p}")
    # de-dupe while preserving order
    seen = set()
    out = []
    for f in files:
        key = f.resolve()
        if key not in seen:
            seen.add(key)
            out.append(f)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert .m4a to .wav")
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="One or more .m4a files, or directories to search recursively",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: same folder as each source file)",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=None,
        help="Resample to this sample rate (e.g. 24000 for StyleTTS2)",
    )
    args = parser.parse_args()

    files = collect_inputs(args.inputs)
    if not files:
        print("No .m4a files found.", file=sys.stderr)
        return 1

    for src in files:
        if args.output is not None:
            dst = args.output / (src.stem + ".wav")
        else:
            dst = src.with_suffix(".wav")
        print(f"{src} -> {dst}")
        convert_one(src, dst, args.sr)

    print(f"Done. Converted {len(files)} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
