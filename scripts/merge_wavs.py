#!/usr/bin/env python3
"""Concatenate .wav files into one .wav (in the order given).

Requires pydub + ffmpeg on PATH for some codecs; plain PCM wav works with pydub alone.

Examples:
  python scripts/merge_wavs.py a.wav b.wav -o merged.wav
  python scripts/merge_wavs.py ./clips/*.wav -o all.wav --sr 24000
  python scripts/merge_wavs.py ./wav_dir -o merged.wav
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pydub import AudioSegment


def collect_wavs(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for p in paths:
        if p.is_dir():
            files.extend(sorted(p.glob("*.wav")))
            files.extend(sorted(p.glob("*.WAV")))
        elif p.is_file():
            if p.suffix.lower() != ".wav":
                raise SystemExit(f"Not a .wav file: {p}")
            files.append(p)
        else:
            raise SystemExit(f"Path not found: {p}")
    seen = set()
    out: list[Path] = []
    for f in files:
        key = f.resolve()
        if key not in seen:
            seen.add(key)
            out.append(f)
    return out


def merge_wavs(sources: list[Path], dst: Path, sr: int | None) -> None:
    if not sources:
        raise SystemExit("No .wav files to merge.")

    clips: list[AudioSegment] = []
    for src in sources:
        print(f"  + {src}")
        audio = AudioSegment.from_file(src, format="wav")
        clips.append(audio)

    merged = clips[0]
    for clip in clips[1:]:
        merged += clip

    if sr is not None:
        merged = merged.set_frame_rate(sr)
    merged = merged.set_channels(1).set_sample_width(2)

    dst = dst.with_suffix(".wav")
    dst.parent.mkdir(parents=True, exist_ok=True)
    merged.export(dst, format="wav")
    print(f"Wrote {dst} ({len(merged) / 1000:.1f}s, {len(sources)} files)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge .wav files into one .wav")
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Two or more .wav files, or directories of .wav files",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("merged.wav"),
        help="Output .wav path (default: merged.wav)",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=None,
        help="Optional resample rate (e.g. 24000)",
    )
    args = parser.parse_args()

    files = collect_wavs(args.inputs)
    if len(files) < 2:
        print("Need at least two .wav files to merge.", file=sys.stderr)
        return 1

    print(f"Merging {len(files)} files -> {args.output}")
    merge_wavs(files, args.output, args.sr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
