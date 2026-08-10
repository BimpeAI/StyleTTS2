# Helper scripts (Bimpe / data prep)

Run these from the **repository root** (paths like `Data/` and `Models/` assume that).

| Script | Purpose |
|--------|---------|
| `prepare_wav_for_training.py` | Long recording → short wavs + `metadata.csv` + IPA lists |
| `prepare_bimpe_data.py` | `metadata.csv` + wavs → phonemized train/val + OOD |
| `datasplit.py` | Split metadata into train/val IPA lists |
| `m4a_to_wav.py` | Convert m4a → 24 kHz wav |
| `cut_and_merge_wav.py` | Trim/merge audio for a reference clip |
| `trim_audio_samples.py` | Export N-minute prefixes from a long wav |
| `numbers_to_words.py` | Normalize currency / digit strings for TTS text |
| `download_libritts_pretrained.sh` | Fetch LibriTTS StyleTTS2 checkpoint → `Models/LibriTTS/` |

Examples:

```bash
bash scripts/download_libritts_pretrained.sh
python scripts/prepare_wav_for_training.py long.wav --out_dataset /path/to/Dataset ...
python scripts/prepare_bimpe_data.py --metadata ... --wavs ... --out_dir ...
```

Training and serving entrypoints stay at the repo root (`train_*.py`, `serving/`).
Docs: [docs/NEW_SPEAKER_FINETUNE.md](../docs/NEW_SPEAKER_FINETUNE.md).
