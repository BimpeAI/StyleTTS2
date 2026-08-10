# StyleTTS 2: Towards Human-Level Text-to-Speech through Style Diffusion and Adversarial Training with Large Speech Language Models

### Yinghao Aaron Li, Cong Han, Vinay S. Raghavan, Gavin Mischler, Nima Mesgarani

> In this paper, we present StyleTTS 2, a text-to-speech (TTS) model that leverages style diffusion and adversarial training with large speech language models (SLMs) to achieve human-level TTS synthesis. StyleTTS 2 differs from its predecessor by modeling styles as a latent random variable through diffusion models to generate the most suitable style for the text without requiring reference speech, achieving efficient latent diffusion while benefiting from the diverse speech synthesis offered by diffusion models. Furthermore, we employ large pre-trained SLMs, such as WavLM, as discriminators with our novel differentiable duration modeling for end-to-end training, resulting in improved speech naturalness. StyleTTS 2 surpasses human recordings on the single-speaker LJSpeech dataset and matches it on the multispeaker VCTK dataset as judged by native English speakers. Moreover, when trained on the LibriTTS dataset, our model outperforms previous publicly available models for zero-shot speaker adaptation. This work achieves the first human-level TTS synthesis on both single and multispeaker datasets, showcasing the potential of style diffusion and adversarial training with large SLMs.

Paper: [https://arxiv.org/abs/2306.07691](https://arxiv.org/abs/2306.07691)

Audio samples: [https://styletts2.github.io/](https://styletts2.github.io/)

Online demo: [Hugging Face](https://huggingface.co/spaces/styletts2/styletts2) (thank [@fakerybakery](https://github.com/fakerybakery) for the wonderful online demo)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yl4579/StyleTTS2/blob/main/) [![Discord](https://img.shields.io/discord/1197679063150637117?logo=discord&logoColor=white&label=Join%20our%20Community)](https://discord.gg/ha8sxdG2K4)

---

## Project overview

This repository is **StyleTTS 2** upstream, plus a **BimpeTTS** finetune path used in this fork:

| Path | Purpose |
|------|---------|
| Upstream LJSpeech / LibriTTS | Train or finetune with [Configs/config.yml](Configs/config.yml), [Configs/config_ft.yml](Configs/config_ft.yml) |
| **BimpeTTS (recommended)** | Finetune LibriTTS pretrained weights on your speaker via [Configs/config_bimpe_ft.yml](Configs/config_bimpe_ft.yml) |
| Inference | [Demo/Inference_LJSpeech.ipynb](Demo/Inference_LJSpeech.ipynb) |
| Production TTS API | [serving/tts_server.py](serving/tts_server.py) |

For a new speaker with limited data, prefer **LibriTTS finetune** over training from scratch.

### Repository layout

| Location | Contents |
|----------|----------|
| Repo root | Upstream StyleTTS2 core (`models.py`, `meldataset.py`, `train_*.py`, …) |
| [Configs/](Configs/) | Training / finetune YAML (incl. `config_bimpe_*.yml`) |
| [Modules/](Modules/), [Utils/](Utils/) | Model modules and ASR / JDC / PLBERT utilities |
| [Demo/](Demo/), [Colab/](Colab/) | Notebooks and conversation demos |
| [serving/](serving/) | FastAPI TTS HTTP server |
| [scripts/](scripts/) | Bimpe data-prep and audio helpers (see [scripts/README.md](scripts/README.md)) |
| [docs/](docs/) | Fork guides (e.g. [docs/NEW_SPEAKER_FINETUNE.md](docs/NEW_SPEAKER_FINETUNE.md)) |
| `Models/`, `Data/`, `voices/` | Checkpoints, lists, and style refs (local; often gitignored) |

---

## Requirements

- **GPU** with CUDA (training and low-latency serving). CPU inference works but is slow.
- **Python** 3.10 or 3.11 recommended (3.7+ upstream).
- **CUDA-matched** `torch` / `torchaudio` (example: torch 2.4.x + cu124).
- **System:** `espeak-ng` (phonemizer backend).
- **Inference / server pins** in [requirements_ljspeech.txt](requirements_ljspeech.txt):
  - `numpy==1.26.4`
  - `transformers==4.51.3`
  - Resemble AI `monotonic_align` (not the plain PyPI package)
  - `fastapi` + `uvicorn` for the TTS HTTP service

### Pretrained Utils (required before train/infer)

Verify these exist under the repo:

- `Utils/ASR/epoch_00080.pth` (+ `Utils/ASR/config.yml`)
- `Utils/JDC/bst.t7`
- `Utils/PLBERT/step_*.t7` (and PLBERT config under `Utils/PLBERT/`)

See [Utils](Utils) and upstream notes below for ASR / JDC / PLBERT sources.

---

## Quick setup

```bash
git clone <this-repo-url>
cd StyleTTS2   # or your clone directory

# 1) Torch (match your CUDA)
pip install torch==2.4.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124

# 2) Training deps (upstream) and/or inference + server deps
pip install -r requirements.txt
pip install -r requirements_ljspeech.txt

# 3) System phonemizer backend (Debian/Ubuntu)
sudo apt-get install -y espeak-ng espeak-ng-data libespeak-ng1
```

On Windows, install a CUDA torch build from the official index after the requirements install if needed.

Quick sanity check:

```bash
ls Utils/ASR/epoch_00080.pth Utils/JDC/bst.t7 Utils/PLBERT/
python -c "import torch, phonemizer; print('cuda:', torch.cuda.is_available())"
```

---

## Upstream training

Data list format: `filename.wav|transcription|speaker` (see [Data/val_list.txt](Data/val_list.txt)). Speaker labels are required for multi-speaker / style-reference sampling.

First stage:

```bash
accelerate launch train_first.py --config_path ./Configs/config.yml
```

Second stage (DP; DDP for `train_second.py` is still broken upstream — see [#7](https://github.com/yl4579/StyleTTS2/issues/7)):

```bash
python train_second.py --config_path ./Configs/config.yml
```

Checkpoints: `epoch_1st_%05d.pth` / `epoch_2nd_%05d.pth` under `log_dir`.

### Important configs ([Configs/config.yml](Configs/config.yml))

- `OOD_data` — out-of-distribution texts for SLM adversarial training (`text|anything`)
- `min_length` / `max_len` — OOD / audio length controls (frames; hop 300 ≈ 12.5 ms)
- `multispeaker` — must match architecture of any pretrained weights you load
- `batch_percentage` — lower if OOM during SLM adversarial training

### Upstream finetune (LJSpeech example)

```bash
python train_finetune.py --config_path ./Configs/config_ft.yml
# or single-GPU accelerate:
accelerate launch --mixed_precision=fp16 --num_processes=1 train_finetune_accelerate.py --config_path ./Configs/config_ft.yml
```

Requires the LibriTTS checkpoint under `Models/LibriTTS/` (see [Finetuning](#finetuning) notes below). Quality tips: [#81](https://github.com/yl4579/StyleTTS2/discussions/81), [#128](https://github.com/yl4579/StyleTTS2/discussions/128).

### Pre-trained modules (Utils)

- **ASR** — text aligner ([yl4579/AuxiliaryASR](https://github.com/yl4579/AuxiliaryASR))
- **JDC** — pitch extractor ([yl4579/PitchExtractor](https://github.com/yl4579/PitchExtractor))
- **PLBERT** — [yl4579/PL-BERT](https://github.com/yl4579/PL-BERT); multilingual option: [papercup-ai/multilingual-pl-bert](https://huggingface.co/papercup-ai/multilingual-pl-bert)

---

## BimpeTTS path (this fork)

Recommended for a single target voice with limited data: **download LibriTTS StyleTTS2 → finetune**, not full from-scratch stage-1/2 on only a few hours of audio.

### 1) Prepare data

Transcript lists must be IPA phonemes:

```text
filename.wav|ipa_phonemes|speaker_id
```

Helpers (run from repo root; see [scripts/README.md](scripts/README.md)):

- [scripts/datasplit.py](scripts/datasplit.py)
- [scripts/prepare_bimpe_data.py](scripts/prepare_bimpe_data.py)
- [scripts/prepare_wav_for_training.py](scripts/prepare_wav_for_training.py) — long WAV → wavs + metadata + IPA lists

Typical remote paths used by the Bimpe configs:

| Role | Path |
|------|------|
| WAVs (24 kHz) | `/root/BimpeTTS_Dataset/wavs` |
| Train / val lists | `/root/Data/train_list.txt`, `/root/Data/val_list.txt` |
| OOD texts | `/root/Data/OOD_texts.txt` |

Adjust paths in [Configs/config_bimpe_ft.yml](Configs/config_bimpe_ft.yml) if your layout differs. Keep one `speaker_id` for a single speaker; do not mix another person’s voice into the same id.

### 2) Download LibriTTS pretrained checkpoint

```bash
bash scripts/download_libritts_pretrained.sh
# → Models/LibriTTS/epochs_2nd_00020.pth
```

### 3) Finetune (recommended)

```bash
python train_finetune.py --config_path Configs/config_bimpe_ft.yml
```

Outputs under `Models/BimpeTTS_ft/` (`epoch_2nd_*.pth`).

Continue from the latest checkpoint:

```bash
python train_finetune.py --config_path Configs/config_bimpe_ft_continue.yml
```

Architecture in the Bimpe configs stays **multispeaker + HiFi-GAN** so it matches the LibriTTS pretrained weights.

### 3b) Specialize on a new speaker (continue finetune)

To prioritize a **new** speaker’s voice (and accept that Bimpe will weaken), follow **[docs/NEW_SPEAKER_FINETUNE.md](docs/NEW_SPEAKER_FINETUNE.md)**. Summary:

1. `scripts/prepare_wav_for_training.py` → new `wavs/` + `metadata.csv` + IPA lists (`speaker_id=0`)
2. Continue from best Bimpe weights on **only** that data (no Bimpe merge):

```bash
python train_finetune.py --config_path Configs/config_bimpe_ft_new_speaker.yml
```

Outputs: `Models/BimpeTTS_ft_new/best_2nd.pth`. Use a short clip from the new speaker as `REF_WAV` at inference.

### 4) From-scratch (optional / usually worse on small data)

```bash
python train_first_bimpe.py --config_path Configs/config_bimpe.yml
python train_second.py --config_path Configs/config_bimpe.yml
```

Expect lower naturalness than LibriTTS finetune when the dataset is small.

### 5) Inference (notebook)

Open [Demo/Inference_LJSpeech.ipynb](Demo/Inference_LJSpeech.ipynb) and set:

- `CONFIG_PATH = 'Configs/config_bimpe_ft.yml'`
- `CHECKPOINT_PATH = None` → auto-picks latest `Models/BimpeTTS_ft/epoch_2nd_*.pth`
- `REF_WAV` → a clean clip of the target speaker (e.g. `/root/BimpeTTS_Dataset/wavs/bimpe_0001.wav`)

Long text is synthesized **sentence by sentence** with style continuity between sentences.

---

## Production: real-time voice conversation

StyleTTS2 in this repo synthesizes **full utterances / sentences**, not true sample-by-sample streaming. For a conversation bot, wrap STT + LLM + this TTS API and optimize turn latency with chunking and overlap.

```mermaid
flowchart LR
  Mic[Mic / Web client] --> VAD[VAD / turn end]
  VAD --> STT[STT]
  STT --> LLM[LLM]
  LLM -->|tokens / sentences| TTS[StyleTTS2 FastAPI]
  TTS -->|WAV chunks| Play[Audio playback]
```

### Latency strategy

1. **VAD** — detect end of user turn before calling STT/LLM.
2. **Stream LLM** — emit tokens as they arrive; buffer into complete sentences.
3. **Sentence-chunk TTS** — `POST /tts` per sentence (matches notebook / server behavior).
4. **Overlap** — start synthesizing sentence *n+1* while playing sentence *n*.
5. **Warm voices** — server loads once and caches style vectors for each pure-name wav under `VOICES_DIR` (skips `*_sample`).

Example stack:

| Component | Options |
|-----------|---------|
| STT | Whisper / faster-whisper, or cloud STT |
| LLM | OpenAI API or local LLM |
| TTS | This repo’s FastAPI service on a GPU |

### Run the TTS API

Install deps from [requirements_ljspeech.txt](requirements_ljspeech.txt) (includes `fastapi` and `uvicorn`). From the **repo root**:

```bash
export CONFIG_PATH=Configs/config_bimpe_ft.yml
export CHECKPOINT_PATH=Models/BimpeTTS_ft/best_2nd.pth   # or omit to use best/latest
export VOICES_DIR=./voices                               # pure-name *.wav (tara.wav, …)
export STYLETTS_DEFAULT_VOICE=tara
export STYLETTS_MAX_CONCURRENT=1
# optional single-file fallback if VOICES_DIR missing default:
# export REF_WAV=/path/to/tara.wav

uvicorn serving.tts_server:app --host 0.0.0.0 --port 8000
```

- `GET /health` — readiness, device, checkpoint, loaded voices
- `GET /voices` — `{"voices":[...], "default":"tara"}`
- `POST /tts` — JSON `{"text": "Hello there.", "voice": "tara"}` → `audio/wav` @ 24 kHz  
  Optional: `alpha`, `beta`, `diffusion_steps`, `embedding_scale`, `pause_ms`, `speed`

For LiveKit / bimpe-agents integration, see that repo’s `STYLETTS_SELFHOST.md`.

### Client loop (pseudocode)

```python
# capture_audio() -> STT -> LLM -> TTS -> play
user_wav = capture_until_vad()
user_text = stt(user_wav)
for sentence in stream_llm_sentences(user_text):
    r = requests.post("http://localhost:8000/tts", json={"text": sentence})
    play_wav(r.content)  # overlap: fetch next while this plays
```

### Ops notes

- Prefer **one GPU process** (model + ref style stay resident).
- Set `CONFIG_PATH` / `CHECKPOINT_PATH` / `REF_WAV` via env (see above).
- Docker: install `espeak-ng`, bake Utils + checkpoint + ref wav into the image or mount them; use GPU runtime.
- For experimental lower-latency **sample streaming**, see the [NeuralVox StyleTTS2 fork](https://github.com/NeuralVox/StyleTTS2) — not wired into this server.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `DTensor` / transformers import errors with torch 2.4 | Pin `transformers==4.51.3` ([requirements_ljspeech.txt](requirements_ljspeech.txt)) |
| NumPy `_center` / binary incompat | Pin `numpy==1.26.4` |
| `mask_from_lens` / monotonic_align | Install Resemble AI fork: `git+https://github.com/resemble-ai/monotonic_align.git` |
| Missing espeak / empty phonemes | `apt-get install espeak-ng` and reinstall `phonemizer` |
| CUDA OOM (train or TTS) | Lower `batch_size` / `max_len`; reduce concurrent `/tts` load |
| NaN during training | Avoid mixed precision in stage 1; try batch size ~16; see [#10](https://github.com/yl4579/StyleTTS2/issues/10) |
| Checkpoint load shape / key errors | Config architecture must match checkpoint (e.g. Bimpe ft → `config_bimpe_ft.yml`, not single-speaker LJ config) |
| Robotic / wrong voice | Better `REF_WAV`; try lower `alpha` / `beta`; prefer finetune ckpt over from-scratch |
| YAML `UnicodeDecodeError` | Open configs with `encoding='utf-8'` |

Upstream finetune OOM after `joint_epoch`: raise `joint_epoch` past `epochs` to skip SLM adversarial training (quality may drop).

High-pitched noise on old GPUs: use a newer GPU or CPU infer ([#13](https://github.com/yl4579/StyleTTS2/issues/13)).

---

## Upstream pretrained models (reference)

- LJSpeech: [yl4579/StyleTTS2-LJSpeech](https://huggingface.co/yl4579/StyleTTS2-LJSpeech/tree/main)
- LibriTTS: [yl4579/StyleTTS2-LibriTTS](https://huggingface.co/yl4579/StyleTTS2-LibriTTS/tree/main)

You can import StyleTTS 2 in other projects; GPL-licensed packaging and experimental streaming also exist in [NeuralVox/StyleTTS2](https://github.com/NeuralVox/StyleTTS2). A [MIT PyPI package](https://pypi.org/project/styletts2/) using gruut is available with known quality tradeoffs.

***Before using these pre-trained models, you agree to inform the listeners that the speech samples are synthesized by the pre-trained models, unless you have the permission to use the voice you synthesize. That is, you agree to only use voices whose speakers grant the permission to have their voice cloned, either directly or by license before making synthesized voices public, or you have to publicly announce that these voices are synthesized if you do not have the permission to use these voices.***

---

## Finetuning

The script is modified from `train_second.py` which uses DP, as DDP does not work for `train_second.py`. See [#7](https://github.com/yl4579/StyleTTS2/issues/7) if you want to help.

```bash
python train_finetune.py --config_path ./Configs/config_ft.yml
```

Please make sure you have the LibriTTS checkpoint downloaded and unzipped under the folder. The default configuration `config_ft.yml` finetunes on LJSpeech with 1 hour of speech data (around 1k samples) for 50 epochs. This took about 4 hours to finish on four NVidia A100. The quality is slightly worse (similar to NaturalSpeech on LJSpeech) than LJSpeech model trained from scratch with 24 hours of speech data, which took around 2.5 days to finish on four A100. The samples can be found at [#65 (comment)](https://github.com/yl4579/StyleTTS2/discussions/65#discussioncomment-7668393).

If you are using a **single GPU** and want to save training speed and VRAM (thank [@korakoe](https://github.com/korakoe) for [#100](https://github.com/yl4579/StyleTTS2/pull/100)):

```bash
accelerate launch --mixed_precision=fp16 --num_processes=1 train_finetune_accelerate.py --config_path ./Configs/config_ft.yml
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yl4579/StyleTTS2/blob/main/Colab/StyleTTS2_Finetune_Demo.ipynb)

---

## TODO (upstream)

- [x] Training and inference demo code for single-speaker models (LJSpeech)
- [x] Test training code for multi-speaker models (VCTK and LibriTTS)
- [x] Finish demo code for multispeaker model and upload pre-trained models
- [x] Add a finetuning script for new speakers with base pre-trained multispeaker models
- [ ] Fix DDP (accelerator) for `train_second.py` **(see [#7](https://github.com/yl4579/StyleTTS2/issues/7))**

---

## References

- [archinetai/audio-diffusion-pytorch](https://github.com/archinetai/audio-diffusion-pytorch)
- [jik876/hifi-gan](https://github.com/jik876/hifi-gan)
- [rishikksh20/iSTFTNet-pytorch](https://github.com/rishikksh20/iSTFTNet-pytorch)
- [nii-yamagishilab/project-NN-Pytorch-scripts/project/01-nsf](https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts/tree/master/project/01-nsf)

## License

Code: MIT License

Pre-Trained Models: Before using these pre-trained models, you agree to inform the listeners that the speech samples are synthesized by the pre-trained models, unless you have the permission to use the voice you synthesize. That is, you agree to only use voices whose speakers grant the permission to have their voice cloned, either directly or by license before making synthesized voices public, or you have to publicly announce that these voices are synthesized if you do not have the permission to use these voices.
