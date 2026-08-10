# New-speaker continue finetune

Specialize the best Bimpe checkpoint on **only** a new speaker (e.g. ~26 min WAV).  
**Do not** merge Bimpe training data — Bimpe identity will weaken by design.

## Prerequisites

```bash
cd /root/StyleTTS2   # or your repo path
pip install openai-whisper soundfile librosa phonemizer
# system: espeak-ng, ffmpeg; GPU recommended
```

Confirm:

- Checkpoint: `Models/BimpeTTS_ft/best_2nd.pth`
- Repo OOD: `Data/OOD_texts.txt`

## Phase 1 — Build `wavs/` + `metadata.csv`

```bash
python prepare_wav_for_training.py /path/to/new_female_26min.wav \
  --out_dataset /root/NewSpeaker_Dataset \
  --out_lists /root/NewSpeaker_Data \
  --prefix female2 \
  --whisper_model medium \
  --language en \
  --phoneme_lang en-us \
  --speaker_id 0 \
  --min_seconds 1.0 \
  --max_seconds 12.0 \
  --val_ratio 0.05
```

| Path | Content |
|------|---------|
| `/root/NewSpeaker_Dataset/wavs/*.wav` | Short 24 kHz clips |
| `/root/NewSpeaker_Dataset/metadata.csv` | `file_name\|text` |
| `/root/NewSpeaker_Data/train_list.txt` | `filename.wav\|ipa\|0` |
| `/root/NewSpeaker_Data/val_list.txt` | same |
| `/root/NewSpeaker_Data/OOD_texts.txt` | copied OOD |

**Quality gate:** Spot-check ~20 `metadata.csv` rows vs audio. After fixing Whisper errors:

```bash
python prepare_bimpe_data.py \
  --metadata /root/NewSpeaker_Dataset/metadata.csv \
  --wavs /root/NewSpeaker_Dataset/wavs \
  --out_dir /root/NewSpeaker_Data \
  --repo_ood Data/OOD_texts.txt \
  --speaker_id 0 \
  --val_ratio 0.05
```

Aim for roughly **80–150+** clips from 26 minutes.

## Phase 2 — Continue finetune (new speaker only)

Config: [Configs/config_bimpe_ft_new_speaker.yml](Configs/config_bimpe_ft_new_speaker.yml)

- Starts from `Models/BimpeTTS_ft/best_2nd.pth`
- Writes to `Models/BimpeTTS_ft_new/` (old Bimpe best stays safe)
- Data: `/root/NewSpeaker_Dataset/wavs` + `/root/NewSpeaker_Data/*`
- Mild LRs, `lambda_sty: 2`, `diff_epoch` / `joint_epoch`: 0

```bash
python train_finetune.py --config_path Configs/config_bimpe_ft_new_speaker.yml
```

Prefer early-stopped `Models/BimpeTTS_ft_new/best_2nd.pth`.

## Phase 3 — Validate

1. Use a **short** ref (~5–15 s) from `female2_*.wav` — not the full 26 min file.
2. Point demos/server at `Models/BimpeTTS_ft_new/best_2nd.pth` + that ref.
3. If still weak: continue again from `Models/BimpeTTS_ft_new/best_2nd.pth` with slightly lower LR; do **not** re-add Bimpe data if this voice is the priority.

## Expectations

- This is a **voice specialization**, not a general upgrade over Bimpe.
- ~26 min moves identity toward the new speaker; do not expect multi-hour naturalness.
- Clean Whisper text + short inference chunks matter more than forcing loss to zero.
