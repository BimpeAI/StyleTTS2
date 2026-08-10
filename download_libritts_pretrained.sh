#!/usr/bin/env bash
# Download LibriTTS StyleTTS2 checkpoint for finetuning on Bimpe.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p Models/LibriTTS
cd Models/LibriTTS

if [[ ! -f epochs_2nd_00020.pth ]]; then
  echo "Downloading LibriTTS StyleTTS2 checkpoint..."
  # Official HF repo: https://huggingface.co/yl4579/StyleTTS2-LibriTTS
  wget -c https://huggingface.co/yl4579/StyleTTS2-LibriTTS/resolve/main/epochs_2nd_00020.pth
else
  echo "Checkpoint already present: Models/LibriTTS/epochs_2nd_00020.pth"
fi

ls -lh epochs_2nd_00020.pth
echo "Next:"
echo "  python train_finetune.py --config_path Configs/config_bimpe_ft.yml"
