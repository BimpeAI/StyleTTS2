#!/usr/bin/env python3
"""Cut time ranges from a WAV/MP4 and merge into one WAV.

MP4 needs ffmpeg on PATH (pydub).

Usage:
  python scripts/cut_and_merge_wav.py input.wav -o output.wav
  python scripts/cut_and_merge_wav.py input.mp4 -o output.wav
  python scripts/cut_and_merge_wav.py input.mp4 -o output.wav --sr 24000
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pydub import AudioSegment

# (start, end) as M:SS / MM:SS (minutes may exceed 59) or H:MM:SS
SEGMENTS = [
    ("1:13", "6:19"),
    ("6:22", "8:00"),
    ("8:02", "8:20"),
    ("8:57", "10:35"),
    ("11:25", "12:05"),
    ("13:25", "17:05"),
    ("17:30", "19:20"),
    ("20:31", "21:34"),
    ("21:45", "29:17"),
    ("29:37", "30:08"),
    ("30:45", "32:05"),
    ("32:40", "34:57"),
    ("35:10", "35:27"),
    ("35:41", "36:32"),
    ("37:50", "43:05"),
    ("44:30", "57:10"),
    ("51:30", "53:24"),
    ("53:30", "54:20"),
    ("54:40", "60:30"),
    ("60:40", "62:22"),
    ("62:35", "63:05"),
    ("63:25", "64:45"),
    ("65:07", "66:40"),
    ("66:50", "67:20"),
    ("68:30", "69:29"),
    ("70:28", "74:06"),
    ("74:55", "76:23"),
    ("76:37", "79:25"),
    ("79:35", "81:44"),
]

SUPPORTED_INPUT = {".wav", ".mp4"}


def parse_timestamp(ts: str) -> int:
    """Convert M:SS or H:MM:SS to milliseconds."""
    parts = [int(p) for p in ts.strip().split(":")]
    if len(parts) == 2:
        minutes, seconds = parts
        hours = 0
    elif len(parts) == 3:
        hours, minutes, seconds = parts
    else:
        raise ValueError(f"Bad timestamp (use M:SS or H:MM:SS): {ts!r}")
    return ((hours * 60 + minutes) * 60 + seconds) * 1000


def load_audio(src: Path) -> AudioSegment:
    suffix = src.suffix.lower()
    if suffix not in SUPPORTED_INPUT:
        raise SystemExit(f"Unsupported input type {suffix!r}; use .wav or .mp4")
    # from_file handles wav and mp4 (ffmpeg for mp4)
    return AudioSegment.from_file(src)


def cut_and_merge(
    src: Path,
    dst: Path,
    segments: list[tuple[str, str]] = SEGMENTS,
    sr: int | None = None,
) -> None:
    audio = load_audio(src)
    clips = []
    for start_s, end_s in segments:
        start_ms = parse_timestamp(start_s)
        end_ms = parse_timestamp(end_s)
        if end_ms <= start_ms:
            raise ValueError(f"End must be after start: {start_s} - {end_s}")
        if start_ms >= len(audio):
            raise ValueError(f"Start {start_s} is past end of file ({len(audio) / 1000:.1f}s)")
        clip = audio[start_ms:end_ms]
        print(f"  {start_s} -> {end_s}  ({len(clip) / 1000:.1f}s)")
        clips.append(clip)

    merged = clips[0]
    for clip in clips[1:]:
        merged += clip

    if sr is not None:
        merged = merged.set_frame_rate(sr)

    dst = dst.with_suffix(".wav")
    dst.parent.mkdir(parents=True, exist_ok=True)
    merged.export(dst, format="wav")
    print(f"Wrote {dst} ({len(merged) / 1000:.1f}s total, {len(clips)} segments)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cut WAV/MP4 segments and merge to WAV")
    parser.add_argument("input", type=Path, help="Source .wav or .mp4 file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("merged_clips2.wav"),
        help="Output .wav path (default: merged_clips.wav)",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=None,
        help="Optional resample rate (e.g. 24000)",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"Input not found: {args.input}")

    print(f"Cutting {args.input} ...")
    cut_and_merge(args.input, args.output, sr=args.sr)


if __name__ == "__main__":
    main()
