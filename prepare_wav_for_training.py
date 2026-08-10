#!/usr/bin/env python3
"""Build StyleTTS2 stage-1/2 training data from a single long .wav.

Pipeline:
  1) Transcribe with Whisper (segment timestamps)
  2) Cut each segment into wavs/<prefix>_XXXX.wav @ 24 kHz mono
  3) Write metadata.csv  ->  filename|transcript
  4) Phonemize + split   ->  train_list.txt / val_list.txt
     format: filename.wav|ipa phonemes|speaker_id
  5) Ensure OOD_texts.txt is available for stage-2 SLM training

Deps:
  pip install openai-whisper soundfile librosa phonemizer
  # plus espeak-ng on the system
  # optional: torch with CUDA for faster Whisper

Example:
  python prepare_wav_for_training.py long_recording.wav \\
    --out_dataset ./BimpeTTS_Dataset \\
    --out_lists ./Data \\
    --whisper_model small

Then point config data_params at:
  root_path:  <out_dataset>/wavs
  train_data: <out_lists>/train_list.txt
  val_data:   <out_lists>/val_list.txt
  OOD_data:   <out_lists>/OOD_texts.txt
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
import random
import shutil
import subprocess
import sys
from pathlib import Path


def _load_audio_24k(path: str):
    import librosa
    import numpy as np

    y, sr = librosa.load(path, sr=24000, mono=True)
    return y.astype(np.float32), 24000


def _write_wav(path: str, audio, sr: int = 24000) -> None:
    import numpy as np
    import soundfile as sf

    audio = np.asarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(audio))) + 1e-8
    if peak > 1.0:
        audio = audio / peak
    sf.write(path, audio, sr, subtype="PCM_16")


def _transcribe_whisper(wav_path: str, model_name: str, language: str | None):
    import whisper

    print(f"Loading Whisper model '{model_name}'...")
    model = whisper.load_model(model_name)
    print(f"Transcribing {wav_path} ...")
    kwargs = {
        "verbose": False,
        "condition_on_previous_text": True,
        "without_timestamps": False,
    }
    if language:
        kwargs["language"] = language
    result = model.transcribe(wav_path, **kwargs)
    segments = []
    for seg in result.get("segments") or []:
        text = (seg.get("text") or "").strip()
        start = float(seg.get("start") or 0.0)
        end = float(seg.get("end") or 0.0)
        if not text or end <= start:
            continue
        segments.append({"start": start, "end": end, "text": text})
    print(f"Whisper returned {len(segments)} segments")
    return segments


def _merge_short_segments(segments, min_seconds: float, max_seconds: float):
    """Merge tiny Whisper chunks; keep under max_seconds when possible."""
    if not segments:
        return []
    merged = []
    cur = dict(segments[0])
    for seg in segments[1:]:
        cur_dur = cur["end"] - cur["start"]
        next_dur = seg["end"] - seg["start"]
        gap = seg["start"] - cur["end"]
        can_merge = (
            cur_dur < min_seconds
            or (cur_dur + next_dur + max(0.0, gap) <= max_seconds and gap < 0.75)
        )
        if can_merge and (cur["end"] + max(next_dur, 0) - cur["start"]) <= max_seconds + 0.5:
            cur["end"] = seg["end"]
            cur["text"] = (cur["text"] + " " + seg["text"]).strip()
        else:
            merged.append(cur)
            cur = dict(seg)
    merged.append(cur)
    return merged


def _phonemize_texts(texts, language="en-us"):
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


def _ensure_ood(repo_ood: str, out_lists: str) -> str:
    if not osp.isfile(repo_ood):
        alt = osp.join(osp.dirname(__file__), "Data", "OOD_texts.txt")
        if osp.isfile(alt):
            repo_ood = alt
    if not osp.isfile(repo_ood):
        raise SystemExit(
            f"Missing OOD file at {repo_ood}. "
            "Copy StyleTTS2 Data/OOD_texts.txt before stage-2 training."
        )
    os.makedirs(out_lists, exist_ok=True)
    dest = osp.join(out_lists, "OOD_texts.txt")
    if osp.abspath(repo_ood) != osp.abspath(dest):
        shutil.copy2(repo_ood, dest)
        print(f"Copied OOD -> {dest}")
    return dest


def main():
    parser = argparse.ArgumentParser(
        description="WAV -> wavs/ + metadata.csv + train/val lists for StyleTTS2"
    )
    parser.add_argument("input_wav", type=Path, help="Source .wav (or any ffmpeg-readable audio)")
    parser.add_argument(
        "--out_dataset",
        type=Path,
        default=Path("BimpeTTS_Dataset"),
        help="Dataset root (creates wavs/ + metadata.csv here)",
    )
    parser.add_argument(
        "--out_lists",
        type=Path,
        default=Path("Data"),
        help="Where to write train_list.txt / val_list.txt / OOD_texts.txt",
    )
    parser.add_argument("--prefix", default="clip", help="Filename prefix for clipped wavs")
    parser.add_argument("--whisper_model", default="small", help="tiny/base/small/medium/large")
    parser.add_argument(
        "--language",
        default="en",
        help="Whisper language code (e.g. en). Empty string = auto-detect",
    )
    parser.add_argument("--phoneme_lang", default="en-us", help="espeak language for phonemizer")
    parser.add_argument("--min_seconds", type=float, default=1.0, help="Drop / merge under this length")
    parser.add_argument("--max_seconds", type=float, default=12.0, help="Preferred max clip length")
    parser.add_argument("--pad_ms", type=int, default=50, help="Pad each cut by this many ms")
    parser.add_argument("--speaker_id", type=int, default=0)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--repo_ood",
        default="Data/OOD_texts.txt",
        help="Source OOD_texts.txt from the StyleTTS2 repo",
    )
    parser.add_argument(
        "--skip_lists",
        action="store_true",
        help="Only write wavs/ + metadata.csv (skip IPA train/val lists)",
    )
    args = parser.parse_args()

    if not args.input_wav.is_file():
        raise SystemExit(f"Input not found: {args.input_wav}")

    # Normalize input to a temp 24k wav if needed (Whisper likes clean wav)
    work_wav = args.input_wav
    suffix = args.input_wav.suffix.lower()
    tmp_wav = None
    if suffix != ".wav":
        tmp_wav = args.out_dataset / "_source_24k.wav"
        args.out_dataset.mkdir(parents=True, exist_ok=True)
        print(f"Converting {args.input_wav} -> {tmp_wav}")
        audio, sr = _load_audio_24k(str(args.input_wav))
        _write_wav(str(tmp_wav), audio, sr)
        work_wav = tmp_wav

    lang = args.language.strip() or None
    segments = _transcribe_whisper(str(work_wav), args.whisper_model, lang)
    segments = _merge_short_segments(segments, args.min_seconds, args.max_seconds)

    audio, sr = _load_audio_24k(str(work_wav))
    wavs_dir = args.out_dataset / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)

    pad = args.pad_ms / 1000.0
    metadata_rows = []
    kept = 0
    skipped = 0

    for i, seg in enumerate(segments, start=1):
        start = max(0.0, seg["start"] - pad)
        end = min(len(audio) / sr, seg["end"] + pad)
        dur = end - start
        text = " ".join(seg["text"].split())
        if dur < args.min_seconds or not text:
            skipped += 1
            continue
        fname = f"{args.prefix}_{i:04d}.wav"
        out_path = wavs_dir / fname
        i0, i1 = int(start * sr), int(end * sr)
        _write_wav(str(out_path), audio[i0:i1], sr)
        metadata_rows.append((fname, text, dur))
        kept += 1

    if kept == 0:
        raise SystemExit("No usable clips produced. Check audio / Whisper output.")

    meta_path = args.out_dataset / "metadata.csv"
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write("file_name|text\n")
        for fname, text, _dur in metadata_rows:
            # pipe-safe: strip pipes from text
            text = text.replace("|", " ")
            f.write(f"{fname}|{text}\n")

    total_dur = sum(d for _, _, d in metadata_rows)
    print(f"Wrote {kept} clips -> {wavs_dir} (skipped {skipped})")
    print(f"Total speech ~{total_dur / 60:.1f} min")
    print(f"Wrote metadata -> {meta_path}")

    if args.skip_lists:
        print("Skipping train/val lists (--skip_lists).")
        print("Next: python prepare_bimpe_data.py --metadata ... --wavs ...")
        return

    ood_path = _ensure_ood(args.repo_ood, str(args.out_lists))

    texts = [t for _, t, _ in metadata_rows]
    print(f"Phonemizing {len(texts)} texts ({args.phoneme_lang})...")
    try:
        phones = _phonemize_texts(texts, language=args.phoneme_lang)
    except Exception as e:
        raise SystemExit(
            f"Phonemizer failed: {e}\n"
            "Install: pip install phonemizer && apt-get install espeak-ng"
        ) from e

    lines = []
    for (fname, _text, _dur), ps in zip(metadata_rows, phones):
        ps = (ps or "").strip()
        if not ps:
            continue
        lines.append(f"{fname}|{ps}|{args.speaker_id}")

    if len(lines) < 5:
        raise SystemExit(f"Too few phonemized lines ({len(lines)}); check espeak / language.")

    random.seed(args.seed)
    random.shuffle(lines)
    n_val = max(1, int(len(lines) * args.val_ratio))
    val_lines = lines[:n_val]
    train_lines = lines[n_val:]
    if not train_lines:
        train_lines, val_lines = val_lines[:-1], val_lines[-1:]

    args.out_lists.mkdir(parents=True, exist_ok=True)
    train_path = args.out_lists / "train_list.txt"
    val_path = args.out_lists / "val_list.txt"
    train_path.write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    val_path.write_text("\n".join(val_lines) + "\n", encoding="utf-8")

    print(f"Wrote {len(train_lines)} train -> {train_path}")
    print(f"Wrote {len(val_lines)} val   -> {val_path}")
    print(f"OOD ready: {ood_path}")
    print("Example:", train_lines[0])
    print()
    print("Point Configs/config_bimpe.yml (or config_bimpe_ft.yml) data_params to:")
    print(f'  train_data: "{train_path.resolve()}"')
    print(f'  val_data: "{val_path.resolve()}"')
    print(f'  root_path: "{wavs_dir.resolve()}"')
    print(f'  OOD_data: "{Path(ood_path).resolve()}"')
    print()
    print("Stage 1:  python train_first_bimpe.py --config_path Configs/config_bimpe.yml")
    print("Stage 2:  python train_second.py --config_path Configs/config_bimpe.yml")
    print("(Or finetune: python train_finetune.py --config_path Configs/config_bimpe_ft.yml)")

    if tmp_wav is not None and tmp_wav.is_file():
        # keep converted source; uncomment to delete:
        # tmp_wav.unlink()
        pass


if __name__ == "__main__":
    main()
