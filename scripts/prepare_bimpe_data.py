#!/usr/bin/env python3
"""Prepare StyleTTS2 lists for BimpeTTS from metadata.csv + wavs.

Creates phonemized train/val lists in the format:
  filename.wav|ipa phonemes|speaker_id

Also ensures OOD_texts.txt exists for the DataLoader (required every sample).

Example (on the VPS):
  python scripts/prepare_bimpe_data.py \\
    --metadata /root/BimpeTTS_Dataset/metadata.csv \\
    --wavs /root/BimpeTTS_Dataset/wavs \\
    --out_dir /root/Data \\
    --repo_ood Data/OOD_texts.txt \\
    --min_seconds 1.0 \\
    --val_ratio 0.05
"""

from __future__ import annotations

import argparse
import csv
import os
import os.path as osp
import random
import shutil
from pathlib import Path


def _read_metadata(path: str):
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        sample = f.read(4096)
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters="|,")
        reader = csv.reader(f, dialect)
        for row in reader:
            if not row:
                continue
            # Common: filename|text  or filename|text|normalized
            if len(row) < 2:
                continue
            fname = row[0].strip()
            text = row[1].strip()
            if not fname or not text:
                continue
            if fname.lower() in {"file_name", "filename", "wav", "path"}:
                continue
            if not fname.endswith(".wav"):
                fname = fname + ".wav"
            rows.append((fname, text))
    return rows


def _duration_seconds(wav_path: str) -> float:
    import soundfile as sf

    info = sf.info(wav_path)
    return float(info.frames) / float(info.samplerate)


def _phonemize_texts(texts, language="en-us"):
    from phonemizer import phonemize

    return phonemize(
        texts,
        language=language,
        backend="espeak",
        strip=True,
        preserve_punctuation=True,
        with_stress=True,
        njobs=1,
    )


def main():
    parser = argparse.ArgumentParser(description="Prepare BimpeTTS StyleTTS2 training lists")
    parser.add_argument("--metadata", required=True, help="Path to metadata.csv")
    parser.add_argument("--wavs", required=True, help="Directory containing .wav files")
    parser.add_argument("--out_dir", default="/root/Data", help="Where to write train/val lists")
    parser.add_argument(
        "--repo_ood",
        default="Data/OOD_texts.txt",
        help="Source OOD_texts.txt from the StyleTTS2 checkout",
    )
    parser.add_argument(
        "--ood_dest",
        default=None,
        help="Optional copy destination for OOD (default: <repo>/Data/OOD_texts.txt kept; "
        "also copies into out_dir if different)",
    )
    parser.add_argument("--ensure_ood", action="store_true", help="Only verify/copy OOD file and exit")
    parser.add_argument("--min_seconds", type=float, default=1.0, help="Drop clips shorter than this")
    parser.add_argument("--val_ratio", type=float, default=0.05, help="Validation split ratio")
    parser.add_argument("--speaker_id", type=int, default=0, help="Speaker id for single-speaker")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--language", default="en-us", help="espeak language for phonemizer")
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # Ensure OOD exists next to training config expectation
    repo_ood = args.repo_ood
    if not osp.isfile(repo_ood):
        # try relative to this file
        alt = osp.join(osp.dirname(__file__), "Data", "OOD_texts.txt")
        if osp.isfile(alt):
            repo_ood = alt

    if not osp.isfile(repo_ood):
        raise SystemExit(
            f"Missing OOD file at {args.repo_ood}. "
            "Copy StyleTTS2 Data/OOD_texts.txt onto the server before training."
        )

    ood_out = args.ood_dest or osp.join(args.out_dir, "OOD_texts.txt")
    if osp.abspath(repo_ood) != osp.abspath(ood_out):
        shutil.copy2(repo_ood, ood_out)
        print(f"Copied OOD texts -> {ood_out}")
    print(f"OOD ready: {repo_ood} ({osp.getsize(repo_ood)} bytes)")

    # Also keep a copy under the StyleTTS2 Data/ path if we are in the repo
    local_data_ood = osp.join(osp.dirname(__file__), "Data", "OOD_texts.txt")
    if not osp.isfile(local_data_ood):
        os.makedirs(osp.dirname(local_data_ood), exist_ok=True)
        shutil.copy2(repo_ood, local_data_ood)
        print(f"Copied OOD texts -> {local_data_ood}")

    if args.ensure_ood:
        return

    rows = _read_metadata(args.metadata)
    if not rows:
        raise SystemExit(f"No rows parsed from {args.metadata}")

    kept = []
    skipped_missing = 0
    skipped_short = 0
    for fname, text in rows:
        wav_path = osp.join(args.wavs, fname)
        if not osp.isfile(wav_path):
            # allow metadata paths that already include subdir
            wav_path2 = osp.join(args.wavs, osp.basename(fname))
            if osp.isfile(wav_path2):
                fname = osp.basename(fname)
                wav_path = wav_path2
            else:
                skipped_missing += 1
                continue
        try:
            dur = _duration_seconds(wav_path)
        except Exception:
            skipped_missing += 1
            continue
        if dur < args.min_seconds:
            skipped_short += 1
            continue
        kept.append((fname, text, dur))

    print(
        f"Kept {len(kept)} clips "
        f"(skipped missing={skipped_missing}, short<{args.min_seconds}s={skipped_short})"
    )
    if len(kept) < 10:
        raise SystemExit("Too few usable clips; check wav paths / min_seconds.")

    texts = [t for _, t, _ in kept]
    print(f"Phonemizing {len(texts)} texts with espeak ({args.language})...")
    phones = _phonemize_texts(texts, language=args.language)
    if isinstance(phones, str):
        phones = [phones]

    lines = []
    for (fname, _text, _dur), ps in zip(kept, phones):
        ps = (ps or "").strip()
        if not ps:
            continue
        lines.append(f"{fname}|{ps}|{args.speaker_id}")

    random.shuffle(lines)
    n_val = max(1, int(len(lines) * args.val_ratio))
    val_lines = lines[:n_val]
    train_lines = lines[n_val:]
    if not train_lines:
        train_lines, val_lines = val_lines[:-1], val_lines[-1:]

    train_path = osp.join(args.out_dir, "train_list.txt")
    val_path = osp.join(args.out_dir, "val_list.txt")
    with open(train_path, "w", encoding="utf-8") as f:
        f.write("\n".join(train_lines) + "\n")
    with open(val_path, "w", encoding="utf-8") as f:
        f.write("\n".join(val_lines) + "\n")

    print(f"Wrote {len(train_lines)} train -> {train_path}")
    print(f"Wrote {len(val_lines)} val   -> {val_path}")
    print("Done. Point Configs/config_bimpe.yml data_params at these files, then run:")
    print("  python train_first_bimpe.py --config_path Configs/config_bimpe.yml")


if __name__ == "__main__":
    main()
