#!/usr/bin/env python3
"""Two-speaker StyleTTS2 conversation demo with light spoken fillers.

Same as voice_conversation.py, but OpenAI scripts include a few natural
filler / emotion words (e.g. well, hmm, oh, you know) — sparingly.

OpenAI Chat is used ONLY to write the dialogue text.
Both voices are synthesized with your StyleTTS2 checkpoint + reference WAVs
(no OpenAI TTS / voice API).

Run from StyleTTS2 repo root:
  export OPENAI_API_KEY=sk-...
  export CHECKPOINT_PATH=Models/BimpeTTS_ft/best_2nd.pth
  export REF_WAV_BIMPE=/root/BimpeTTS_Dataset/wavs/bimpe_0001.wav
  export REF_WAV_ASSISTANT=/path/to/other_speaker_ref.wav

  python Demo/filler_voice_conversation.py -o Demo/faith_james_friends_chat.wav

Optional:
  --turns 24 --topic "two friends catching up"
  --alpha 0.1 --beta 0.3 --diffusion_steps 10 --speed 1.15
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path

import librosa
import numpy as np
import torch
import torchaudio
import yaml


def find_repo_root() -> Path:
    """Support script in repo root or Demo/."""
    here = Path(__file__).resolve().parent
    for candidate in (here, *here.parents):
        if (candidate / "models.py").is_file() and (candidate / "Configs").is_dir():
            return candidate
    return Path(__file__).resolve().parents[1]


REPO_ROOT = find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)
print(f"Repo root: {REPO_ROOT}", flush=True)

from models import *  # noqa: E402,F401,F403
from utils import *  # noqa: E402,F401,F403
from text_utils import TextCleaner  # noqa: E402
from Modules.diffusion.sampler import (  # noqa: E402
    ADPM2Sampler,
    DiffusionSampler,
    KarrasSchedule,
)

SR = 24000
MEAN, STD = -4, 4
SPEAKERS = ("FAITH", "JAMES")
to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300
)
textcleaner = TextCleaner()


def env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, default)


def find_best_or_latest(log_dir: str) -> str:
    best = os.path.join(log_dir, "best_2nd.pth")
    if os.path.isfile(best):
        return best
    paths = sorted(glob.glob(os.path.join(log_dir, "epoch_2nd_*.pth")))
    if not paths:
        raise FileNotFoundError(f"No checkpoint under {log_dir}")
    return paths[-1]


def length_to_mask(lengths: torch.Tensor) -> torch.Tensor:
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    return torch.gt(mask + 1, lengths.unsqueeze(1))


def preprocess(wave: np.ndarray) -> torch.Tensor:
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    return (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - MEAN) / STD


def load_checkpoint_weights(model, path: str):
    params_whole = torch.load(path, map_location="cpu")
    params = params_whole["net"]
    for key in model:
        if key not in params:
            continue
        state_dict = params[key]
        try:
            model[key].load_state_dict(state_dict, strict=True)
        except Exception:
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_state_dict[k[7:] if k.startswith("module.") else k] = v
            model[key].load_state_dict(new_state_dict, strict=False)
    _ = [model[key].eval() for key in model]
    return params_whole


def resolve_audio_path(path: str) -> str:
    """If path is missing/corrupt, try same stem with common media extensions."""
    path = str(path)
    candidates = [path]
    stem = Path(path).with_suffix("")
    for ext in (".wav", ".WAV", ".m4a", ".mp3", ".flac", ".ogg", ".aac", ".webm", ".mp4"):
        alt = f"{stem}{ext}"
        if alt not in candidates:
            candidates.append(alt)

    last_err = None
    for cand in candidates:
        if not os.path.isfile(cand):
            continue
        try:
            _ = load_audio_mono_24k(cand)
            if cand != path:
                print(f"Note: using alternate audio file -> {cand}", flush=True)
            return cand
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(
        f"No decodable audio found for {path!r}.\n"
        f"Last error: {last_err}\n\n"
        f"Your assistant ref env is probably still set to a broken file.\n"
        f"Fix with a working wav, e.g.:\n"
        f"  unset REF_WAV_OPENAI REF_WAV_ASSISTANT\n"
        f"  export REF_WAV_BIMPE=/root/samples/reference_elisha.wav\n"
        f"  export REF_WAV_ASSISTANT=/root/samples/SOME_OTHER_GOOD.wav\n"
        f"  # temporary same-voice test:\n"
        f"  export REF_WAV_ASSISTANT=/root/samples/reference_elisha.wav\n"
        f"  python3 voice_conversation.py -o Demo/bimpe_conversation.wav\n"
        f"Or pass flags:\n"
        f"  python3 voice_conversation.py --ref-assistant /path/to/good.wav -o Demo/out.wav"
    )


def load_audio_mono_24k(path: str) -> np.ndarray:
    """Load any common audio file as mono float32 @ 24 kHz."""
    path = str(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Audio not found: {path}")

    errors = []

    try:
        wav, sr = torchaudio.load(path)
        if wav.ndim > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != SR:
            wav = torchaudio.functional.resample(wav, sr, SR)
        audio = wav.squeeze(0).numpy().astype(np.float32)
        if audio.size > 0:
            return audio
    except Exception as e:
        errors.append(f"torchaudio: {e}")

    try:
        audio, _ = librosa.load(path, sr=SR, mono=True)
        audio = np.asarray(audio, dtype=np.float32)
        if audio.size > 0:
            return audio
    except Exception as e:
        errors.append(f"librosa: {e}")

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            cmd = [
                ffmpeg, "-y", "-i", path,
                "-ac", "1", "-ar", str(SR), "-sample_fmt", "s16", tmp_path,
            ]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if proc.returncode == 0 and os.path.isfile(tmp_path):
                try:
                    wav, sr = torchaudio.load(tmp_path)
                    if wav.ndim > 1:
                        wav = wav.mean(dim=0, keepdim=True)
                    if sr != SR:
                        wav = torchaudio.functional.resample(wav, sr, SR)
                    audio = wav.squeeze(0).numpy().astype(np.float32)
                    if audio.size > 0:
                        return audio
                finally:
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass
            else:
                err = (proc.stderr or b"").decode("utf-8", errors="ignore")[-400:]
                errors.append(f"ffmpeg: exit={proc.returncode} {err}")
        except Exception as e:
            errors.append(f"ffmpeg: {e}")
    else:
        errors.append("ffmpeg: not installed on PATH")

    raise RuntimeError(
        f"Could not decode reference audio: {path}\n"
        f"File looks corrupt/not audio (ffmpeg also failed). Check with:\n"
        f"  ls -lh {path!r}\n"
        f"  file {path!r}\n"
        f"  ffprobe -hide_banner {path!r}\n"
        f"Re-export or convert from the original source, e.g.:\n"
        f"  ffmpeg -y -i /path/to/original.m4a -ac 1 -ar 24000 /root/samples/reference_tara2_24k.wav\n"
        f"Then:\n"
        f"  export REF_WAV_ASSISTANT=/root/samples/reference_tara2_24k.wav\n"
        f"Details: {'; '.join(errors)}"
    )


class StyleTTSRuntime:
    def __init__(self, config_path: str, checkpoint_path: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        import phonemizer
        from Utils.PLBERT.util import load_plbert

        text_aligner = load_ASR_models(self.config.get("ASR_path"), self.config.get("ASR_config"))
        pitch_extractor = load_F0_models(self.config.get("F0_path"))
        plbert = load_plbert(self.config.get("PLBERT_dir"))
        self.model_params = recursive_munch(self.config["model_params"])
        self.model = build_model(self.model_params, text_aligner, pitch_extractor, plbert)
        _ = [self.model[key].eval() for key in self.model]
        _ = [self.model[key].to(self.device) for key in self.model]
        load_checkpoint_weights(self.model, checkpoint_path)
        self.sampler = DiffusionSampler(
            self.model.diffusion.diffusion,
            sampler=ADPM2Sampler(),
            sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
            clamp=False,
        )
        self.phonemizer = phonemizer.backend.EspeakBackend(
            language="en-us", preserve_punctuation=True, with_stress=True
        )
        print(f"Loaded StyleTTS2 on {self.device}: {checkpoint_path}", flush=True)

    def compute_style(self, path: str) -> torch.Tensor:
        wave = load_audio_mono_24k(path)
        audio, _ = librosa.effects.trim(wave, top_db=30)
        if audio.size < SR // 10:
            audio = wave
        mel_tensor = preprocess(audio.astype(np.float32)).to(self.device)
        with torch.no_grad():
            ref_s = self.model.style_encoder(mel_tensor.unsqueeze(1))
            ref_p = self.model.predictor_encoder(mel_tensor.unsqueeze(1))
        return torch.cat([ref_s, ref_p], dim=1)

    def synthesize_sentence(
        self,
        text: str,
        ref_s: torch.Tensor,
        s_prev=None,
        alpha: float = 0.1,
        beta: float = 0.3,
        t: float = 0.7,
        diffusion_steps: int = 10,
        embedding_scale: float = 1.0,
        speed: float = 1.0,
        max_tokens: int = 500,
    ):
        text = text.strip().replace('"', "")
        ps = self.phonemizer.phonemize([text])
        ps = " ".join(ps[0].split()).replace("``", '"').replace("''", '"')
        tokens = textcleaner(ps)
        tokens.insert(0, 0)
        if len(tokens) > max_tokens:
            raise ValueError(f"Sentence too long ({len(tokens)} tokens): {text[:80]}...")
        if speed <= 0:
            raise ValueError(f"speed must be > 0, got {speed}")

        tokens = torch.LongTensor(tokens).to(self.device).unsqueeze(0)
        model = self.model
        model_params = self.model_params

        with torch.no_grad():
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(self.device)
            text_mask = length_to_mask(input_lengths).to(self.device)
            t_en = model.text_encoder(tokens, input_lengths, text_mask)
            bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
            d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

            s_pred = self.sampler(
                noise=torch.randn((1, 256)).unsqueeze(1).to(self.device),
                embedding=bert_dur,
                embedding_scale=embedding_scale,
                features=ref_s,
                num_steps=diffusion_steps,
            ).squeeze(1)

            if s_prev is not None:
                s_pred = t * s_prev + (1 - t) * s_pred

            s = s_pred[:, 128:]
            ref = s_pred[:, :128]
            ref = alpha * ref + (1 - alpha) * ref_s[:, :128]
            s = beta * s + (1 - beta) * ref_s[:, 128:]
            s_pred = torch.cat([ref, s], dim=-1)

            d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
            x, _ = model.predictor.lstm(d)
            duration = torch.sigmoid(model.predictor.duration_proj(x)).sum(axis=-1)
            # speed > 1 => faster speech (shorter phoneme durations)
            pred_dur = torch.round(duration.squeeze() / speed).clamp(min=1)

            pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().item()))
            c_frame = 0
            for i in range(pred_aln_trg.size(0)):
                pred_aln_trg[i, c_frame : c_frame + int(pred_dur[i].item())] = 1
                c_frame += int(pred_dur[i].item())

            en = d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(self.device)
            if model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(en)
                asr_new[:, :, 0] = en[:, :, 0]
                asr_new[:, :, 1:] = en[:, :, 0:-1]
                en = asr_new

            F0_pred, N_pred = model.predictor.F0Ntrain(en, s)
            asr = t_en @ pred_aln_trg.unsqueeze(0).to(self.device)
            if model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(asr)
                asr_new[:, :, 0] = asr[:, :, 0]
                asr_new[:, :, 1:] = asr[:, :, 0:-1]
                asr = asr_new

            out = model.decoder(asr, F0_pred, N_pred, ref.squeeze().unsqueeze(0))

        return out.squeeze().cpu().numpy()[..., :-50], s_pred

    def synthesize_text(
        self,
        text: str,
        ref_s: torch.Tensor,
        alpha: float = 0.1,
        beta: float = 0.3,
        diffusion_steps: int = 10,
        embedding_scale: float = 1.0,
        pause_ms: int = 100,
        speed: float = 1.0,
    ) -> np.ndarray:
        text = text.strip().replace('"', "")
        parts = re.split(r"(?<=[.!?])\s+", text)
        sentences = []
        for p in parts:
            p = p.strip()
            if not p:
                continue
            if p[-1] not in ".!?":
                p += "."
            sentences.append(p)
        if not sentences:
            raise ValueError("Empty utterance")

        wavs = []
        s_prev = None
        silence = np.zeros(int(SR * pause_ms / 1000.0), dtype=np.float32)
        for i, sent in enumerate(sentences):
            wav, s_prev = self.synthesize_sentence(
                sent,
                ref_s,
                s_prev=s_prev,
                alpha=alpha,
                beta=beta,
                diffusion_steps=diffusion_steps,
                embedding_scale=embedding_scale,
                speed=speed,
            )
            wavs.append(wav.astype(np.float32))
            if i < len(sentences) - 1:
                wavs.append(silence)
        return np.concatenate(wavs)


def _normalize_speaker(raw: str) -> str | None:
    s = str(raw).strip().upper()
    if s in SPEAKERS:
        return s
    # Tolerate old Chat labels / aliases — still mapped to StyleTTS2 refs only
    if s in {
        "OPENAI",
        "AI",
        "ASSISTANT",
        "ASSISTANT_AI",
        "BOT",
        "SPEAKER_A",
        "A",
        "AGENT",
        "SANDRA",
    }:
        return "FAITH"
    if s in {
        "BIMPE",
        "BIMPEAI",
        "BIMPE_AI",
        "SPEAKER_B",
        "B",
        "HUMAN",
        "CUSTOMER",
        "ANDREW",
    }:
        return "JAMES"
    return None


# --- number → word helpers (inlined; requires: pip install num2words) ---
_CURRENCY_RE = re.compile(r"\$\s*([\d,]+(?:\.\d{1,2})?)")
_DIGITS_RE = re.compile(r"\d+")


def _parse_amount(raw: str) -> float:
    return float(raw.replace(",", ""))


def _currency_to_words(raw: str) -> str:
    """Convert '10,000' / '10,000.50' -> 'ten thousand dollars'."""
    try:
        from num2words import num2words
    except ImportError as e:
        raise SystemExit("Install num2words: pip install num2words") from e

    amount = _parse_amount(raw)
    dollars = int(amount)
    cents = int(round((amount - dollars) * 100))

    dollar_words = num2words(dollars, lang="en").replace("-", " ").replace(",", "").lower()
    unit = "dollar" if dollars == 1 else "dollars"
    out = f"{dollar_words} {unit}"

    if cents:
        cent_words = num2words(cents, lang="en").replace("-", " ").replace(",", "").lower()
        cent_unit = "cent" if cents == 1 else "cents"
        out = f"{out} and {cent_words} {cent_unit}"

    return out


def _digits_to_words(raw: str) -> str:
    """Convert '1234567' -> 'one two three four five six seven'."""
    try:
        from num2words import num2words
    except ImportError as e:
        raise SystemExit("Install num2words: pip install num2words") from e
    return " ".join(num2words(int(ch), lang="en") for ch in raw)


def numbers_to_words(text: str) -> str:
    """Replace $amounts and bare digit runs with spoken words for TTS."""

    def _currency_sub(match: re.Match[str]) -> str:
        return _currency_to_words(match.group(1))

    text = _CURRENCY_RE.sub(_currency_sub, text)

    def _digits_sub(match: re.Match[str]) -> str:
        return _digits_to_words(match.group(0))

    return _DIGITS_RE.sub(_digits_sub, text)


def generate_dialogue(topic: str, turns: int, model: str = "gpt-4o-mini") -> list[dict]:
    """Use OpenAI Chat ONLY for text. Voices come from StyleTTS2 + ref wavs."""
    try:
        from openai import OpenAI
    except ImportError as e:
        raise SystemExit(
            "Install openai: pip install openai\nAnd set OPENAI_API_KEY."
        ) from e

    if not env("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY in the environment.")

    client = OpenAI()
    system = (
        "You write casual spoken dialogue scripts for a TTS demo between two friends. "
        "Return ONLY valid JSON with key \"turns\": a list of objects with keys "
        '"speaker" (either "FAITH" or "JAMES") and "text" (one spoken sentence). '
        "FAITH and JAMES are close friends catching up on a phone call — warm, playful, "
        "relaxed, checking on each other (work, weekends, family, plans). "
        "NOT a bank call, NOT customer service. No agent scripts. "
        "Keep the chat flowing: how are you, what have you been up to, plans, "
        "shared jokes, and a friendly wrap-up. "
        "Include MANY mixed casual numbers across the conversation: "
        "money MUST use dollar notation like $25, $120, or $1,500; "
        "also mention times, ages, scores, distances, quantities, and schedules. "
        "For non-money quantities (minutes, hours, days, ages, percent, scores), "
        "write them already in English words "
        "(e.g. thirty minutes, two weeks, seventy five points) when not using $. "
        "At least one-third of the turns should mention a number, dollar amount, "
        "time, or quantity. "
        "Mention both names naturally a few times. "
        "Alternate speakers. Start with FAITH. "
        "Each line under 30 words. No stage directions. "
        "Add spoken fillers to show emotion and natural hesitation "
        "(examples: well, hmm, oh, ah, you know, I see, right, uh, um, "
        "actually, honestly, okay so, let me see). "
        "Use roughly one short filler in most lines (about 2 out of every 3 turns). "
        "Either friend may use fillers; keep it natural between mates. "
        "Do not stack more than two fillers in one sentence, and keep speech clear. "
        "Do not mention OpenAI, ChatGPT, TTS, banks, loans, restaurants, airlines, or AI models."
    )
    user = (
        f"Write a natural {turns}-turn phone conversation about: {topic}. "
        f"Characters: Faith and James — two friends checking up on each other. "
        f"Tone: casual, warm, friend-to-friend (not formal, not a business call). "
        f"Lean into mixed number-heavy chat (prices, plans, times, scores, quantities). "
        f"Include noticeably more spoken fillers for emotion "
        f"(most lines should have one, still natural — not cartoonish). "
        f"Exactly {turns} objects in the turns array."
    )
    resp = client.chat.completions.create(
        model=model,
        temperature=0.9,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    raw = resp.choices[0].message.content or "{}"
    data = json.loads(raw)
    if isinstance(data, list):
        turns_list = data
    elif isinstance(data, dict):
        turns_list = data.get("turns") or data.get("dialogue") or data.get("conversation") or []
    else:
        turns_list = []

    cleaned = []
    for item in turns_list:
        speaker = _normalize_speaker(item.get("speaker", ""))
        text = str(item.get("text", "")).strip()
        if not speaker or not text:
            continue
        # Expand $amounts / digits to words before TTS
        text = numbers_to_words(text)
        cleaned.append({"speaker": speaker, "text": text})
    if len(cleaned) < 2:
        raise RuntimeError(f"Chat returned too few usable turns: {raw[:400]}")
    return cleaned[:turns]


def save_wav(path: Path, audio: np.ndarray, sr: int = SR) -> None:
    audio = np.asarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(audio))) + 1e-8
    if peak > 1.0:
        audio = audio / peak
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), torch.from_numpy(audio).unsqueeze(0), sr)
    print(f"Wrote {path} ({len(audio) / sr:.1f}s)", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Chat-scripted dialogue with light fillers; "
            "BOTH voices via StyleTTS2 + reference WAVs"
        )
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("Demo/faith_james_friends_chat.wav"),
    )
    parser.add_argument(
        "--topic",
        default=(
            "Faith and James catching up as friends: "
            "check how each other is doing, weekend plans, work, "
            "casual chat with mixed everyday numbers "
            "(prices, times, scores, quantities), and a warm goodbye"
        ),
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=24,
        help="Number of dialogue turns (default 24 for a longer friend chat)",
    )
    parser.add_argument("--chat-model", default="gpt-4o-mini", help="OpenAI Chat model for script only")
    parser.add_argument("--config", default=env("CONFIG_PATH", "Configs/config_bimpe_ft.yml"))
    parser.add_argument(
        "--checkpoint",
        default=env("CHECKPOINT_PATH")
        or (
            "Models/BimpeTTS_ft/best_2nd.pth"
            if os.path.isfile("Models/BimpeTTS_ft/best_2nd.pth")
            else None
        ),
    )
    parser.add_argument(
        "--ref-bimpe",
        default=env("REF_WAV_BIMPE", env("REF_WAV", "/root/BimpeTTS_Dataset/wavs/bimpe_0001.wav")),
        help="Reference speaker WAV for JAMES (friend) lines (StyleTTS2)",
    )
    parser.add_argument(
        "--ref-assistant",
        default=env(
            "REF_WAV_ASSISTANT",
            env("REF_WAV_OPENAI", "/root/BimpeTTS_Dataset/wavs/bimpe_0002.wav"),
        ),
        help="Reference speaker WAV for FAITH (friend) lines (StyleTTS2). Alias: REF_WAV_OPENAI",
    )
    parser.add_argument("--alpha", type=float, default=0)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--diffusion-steps", type=int, default=5)
    parser.add_argument(
        "--speed",
        type=float,
        default=1.3,
        help="Speaking rate via duration scaling (>1 faster, <1 slower)",
    )
    parser.add_argument("--gap-ms", type=int, default=50, help="Silence between speakers")
    parser.add_argument("--script-json", type=Path, default=None, help="Optional: save dialogue JSON")
    args = parser.parse_args()

    if not args.checkpoint:
        args.checkpoint = find_best_or_latest("Models/BimpeTTS_ft")

    for label, path in [
        ("config", args.config),
        ("checkpoint", args.checkpoint),
    ]:
        if not os.path.isfile(path):
            raise SystemExit(f"Missing {label}: {path}")

    try:
        args.ref_bimpe = resolve_audio_path(args.ref_bimpe)
        args.ref_assistant = resolve_audio_path(args.ref_assistant)
    except Exception as e:
        raise SystemExit(str(e)) from e

    print("Voice backend: StyleTTS2 only (no OpenAI TTS)", flush=True)
    print(f"REF_WAV_BIMPE     : {args.ref_bimpe}", flush=True)
    print(f"REF_WAV_ASSISTANT : {args.ref_assistant}", flush=True)
    for label, path in [("ref-bimpe", args.ref_bimpe), ("ref-assistant", args.ref_assistant)]:
        audio = load_audio_mono_24k(path)
        print(f"  OK {label}: {len(audio) / SR:.2f}s @ {SR} Hz", flush=True)

    print(
        "Generating dialogue text with OpenAI Chat (script + light fillers)...",
        flush=True,
    )
    dialogue = generate_dialogue(args.topic, args.turns, model=args.chat_model)
    for i, turn in enumerate(dialogue, 1):
        print(f"  [{i}] {turn['speaker']}: {turn['text']}", flush=True)

    if args.script_json:
        args.script_json.parent.mkdir(parents=True, exist_ok=True)
        args.script_json.write_text(json.dumps(dialogue, indent=2), encoding="utf-8")
        print(f"Saved script -> {args.script_json}", flush=True)

    print("Loading StyleTTS2...", flush=True)
    tts = StyleTTSRuntime(args.config, args.checkpoint)
    styles = {
        "JAMES": tts.compute_style(args.ref_bimpe),
        "FAITH": tts.compute_style(args.ref_assistant),
    }

    gap = np.zeros(int(SR * args.gap_ms / 1000.0), dtype=np.float32)
    chunks: list[np.ndarray] = []
    for i, turn in enumerate(dialogue):
        speaker = turn["speaker"]
        print(f"Synthesizing {speaker} with StyleTTS2 + ref wav...", flush=True)
        wav = tts.synthesize_text(
            turn["text"],
            styles[speaker],
            alpha=args.alpha,
            beta=args.beta,
            diffusion_steps=args.diffusion_steps,
            speed=args.speed,
        )
        chunks.append(wav.astype(np.float32))
        if i < len(dialogue) - 1:
            chunks.append(gap)

    save_wav(args.output, np.concatenate(chunks))
    transcript_path = args.output.with_suffix(".txt")
    transcript_path.write_text(
        "\n".join(f"{t['speaker']}: {t['text']}" for t in dialogue) + "\n",
        encoding="utf-8",
    )
    print(f"Transcript -> {transcript_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
