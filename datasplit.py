#!/usr/bin/env python3
"""Split BimpeTTS metadata into StyleTTS2 train/val lists.

Output format (required by StyleTTS2):
  filename.wav|ipa phonemes|speaker_id

NOT absolute paths, NOT plain English text.

Example:
  python datasplit.py
  python datasplit.py --no_phonemize   # debug only; do not train on this
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
import random


def read_metadata(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) < 2:
                continue
            file_id = parts[0].strip()
            text = parts[1].strip()
            if not file_id or not text:
                continue
            if file_id.lower() in {"file_name", "filename", "wav", "path"}:
                continue
            if not file_id.endswith(".wav"):
                file_id = f"{file_id}.wav"
            rows.append((file_id, text))
    return rows


def phonemize_batch(texts, language="en-us"):
    from phonemizer import phonemize

    phones = phonemize(
        texts,
        language=language,
        backend="espeak",
        strip=True,
        preserve_punctuation=True,
        with_stress=True,
        njobs=1,
    )
    if isinstance(phones, str):
        phones = [phones]
    return phones


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", default="/root/BimpeTTS_Dataset/wavs")
    parser.add_argument("--metadata", default="/root/BimpeTTS_Dataset/metadata.csv")
    parser.add_argument("--out_dir", default="/root/Data")
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--speaker_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--language", default="en-us")
    parser.add_argument(
        "--no_phonemize",
        action="store_true",
        help="Keep plain English (NOT valid for StyleTTS2 training)",
    )
    parser.add_argument("--min_seconds", type=float, default=1.0)
    args = parser.parse_args()

    if not osp.isfile(args.metadata):
        raise SystemExit(f"Missing metadata: {args.metadata}")

    try:
        import soundfile as sf
    except ImportError:
        sf = None

    rows = read_metadata(args.metadata)
    kept = []
    skipped = 0
    for fname, text in rows:
        wav_path = osp.join(args.audio_dir, fname)
        if not osp.isfile(wav_path):
            skipped += 1
            continue
        if sf is not None and args.min_seconds > 0:
            try:
                info = sf.info(wav_path)
                dur = float(info.frames) / float(info.samplerate)
                if dur < args.min_seconds:
                    skipped += 1
                    continue
            except Exception:
                skipped += 1
                continue
        kept.append((fname, text))

    print(f"Usable clips: {len(kept)} (skipped {skipped})")
    if len(kept) < 10:
        raise SystemExit("Too few clips. Check --audio_dir / metadata filenames.")

    texts = [t for _, t in kept]
    if args.no_phonemize:
        print("WARNING: writing plain English. Do not train StyleTTS2 on this.")
        contents = texts
    else:
        print(f"Phonemizing {len(texts)} texts...")
        contents = phonemize_batch(texts, language=args.language)

    # Filename only (relative to config root_path), phonemes, speaker
    lines = []
    for (fname, _), content in zip(kept, contents):
        content = (content or "").strip()
        if not content:
            continue
        lines.append(f"{fname}|{content}|{args.speaker_id}")

    random.seed(args.seed)
    random.shuffle(lines)

    split_idx = int(len(lines) * args.train_ratio)
    train_lines = lines[:split_idx]
    val_lines = lines[split_idx:]
    if not val_lines:
        val_lines = train_lines[-max(1, len(train_lines) // 20) :]
        train_lines = train_lines[: -len(val_lines)]

    os.makedirs(args.out_dir, exist_ok=True)
    train_path = osp.join(args.out_dir, "train_list.txt")
    val_path = osp.join(args.out_dir, "val_list.txt")

    with open(train_path, "w", encoding="utf-8") as f:
        f.write("\n".join(train_lines) + "\n")
    with open(val_path, "w", encoding="utf-8") as f:
        f.write("\n".join(val_lines) + "\n")

    print(f"Wrote {len(train_lines)} -> {train_path}")
    print(f"Wrote {len(val_lines)} -> {val_path}")
    print("Example:", train_lines[0])
    print("Config should use:")
    print(f"  train_data: \"{train_path}\"")
    print(f"  val_data: \"{val_path}\"")
    print(f"  root_path: \"{args.audio_dir}\"")


if __name__ == "__main__":
    main()
