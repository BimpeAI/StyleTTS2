#!/usr/bin/env python3
"""Two-speaker StyleTTS2 conversation demo (upgraded synthesis).

Improvements over voice_conversation.py:
  - Long utterances split by phoneme token budget (ALBERT 512 cap), not per-sentence pauses
  - Style continuity (s_prev) across chunks within each turn
  - Trim decoder end-click + cosine crossfade between chunks (no audible chunk joins)
  - Reference speaker WAV paths configured in-code below (not CLI flags)

OpenAI Chat writes dialogue text only; both voices use StyleTTS2 + your ref wavs.

Run from StyleTTS2 repo root:
  export OPENAI_API_KEY=sk-...
  python Demo/voice_conversation_upgrade.py
"""

from __future__ import annotations

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
from nltk.tokenize import word_tokenize

# ---------------------------------------------------------------------------
# Configuration — edit paths and synthesis knobs here
# ---------------------------------------------------------------------------
REF_WAV_FAITH = "/root/samples/tara.wav"  # FAITH (agent) reference
REF_WAV_JAMES = "/root/samples/james.wav"  # JAMES (customer) reference

SPEAKER_REFS: dict[str, str] = {
    "FAITH": REF_WAV_FAITH,
    "JAMES": REF_WAV_JAMES,
}

CONFIG_PATH = "Configs/config_bimpe_ft.yml"
CHECKPOINT_PATH = "Models/BimpeTTS_ft/best_2nd.pth"
OUTPUT_PATH = Path("Demo/verca_bank_loan_inquiry_upgrade.wav")
SCRIPT_JSON_PATH: Path | None = Path("Demo/verca_bank_loan_inquiry_upgrade.json")

TOPIC = (
    "Verca Bank loan inquiry call: Faith greets James, learns he wants a loan, "
    "explains eligibility basics and required documents, outlines how to proceed "
    "with an application, and closes politely"
)
TURNS = 14
CHAT_MODEL = "gpt-4o-mini"

ALPHA = 0.0
BETA = 0.2
DIFFUSION_STEPS = 5
SPEED = 1.3
STYLE_BLEND_T = 0.7  # s_prev weight for prosody continuity across chunks

MAX_TOKENS = 500  # ALBERT limit 512; inference prepends a 0 token
CROSSFADE_MS = 40
TRIM_TAIL_MS = 8
TRIM_HEAD_MS = 4
GAP_MS = 50  # short pause between dialogue turns (different speakers)
# ---------------------------------------------------------------------------


def find_repo_root() -> Path:
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


def trim_pulse(
    w: np.ndarray,
    tail_ms: int = TRIM_TAIL_MS,
    head_ms: int = TRIM_HEAD_MS,
    sr: int = SR,
) -> np.ndarray:
    """Drop StyleTTS decoder end-click and leading noise."""
    w = np.asarray(w, dtype=np.float32).reshape(-1)
    n_tail = int(sr * tail_ms / 1000)
    n_head = int(sr * head_ms / 1000)
    if w.size <= n_tail + n_head + 1:
        return w
    return w[n_head:-n_tail]


def crossfade(
    a: np.ndarray,
    b: np.ndarray,
    fade_ms: int = CROSSFADE_MS,
    sr: int = SR,
) -> np.ndarray:
    """Overlap-add with cosine fade — no inserted silence."""
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    n = min(int(sr * fade_ms / 1000), len(a) // 4, len(b) // 4)
    if n < 8:
        return np.concatenate([a, b])
    fade_out = 0.5 * (1.0 + np.cos(np.linspace(0, np.pi, n, dtype=np.float32)))
    fade_in = fade_out[::-1]
    return np.concatenate([a[:-n], a[-n:] * fade_out + b[:n] * fade_in, b[n:]])


def split_sentences(text: str) -> list[str]:
    text = text.strip().replace('"', "")
    parts = re.split(r"(?<=[.!?])\s+", text)
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if p[-1] not in ".!?":
            p += "."
        out.append(p)
    return out


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
    raise RuntimeError(f"No decodable audio for {path!r}. Last error: {last_err}")


def load_audio_mono_24k(path: str) -> np.ndarray:
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
            cmd = [ffmpeg, "-y", "-i", path, "-ac", "1", "-ar", str(SR), "-sample_fmt", "s16", tmp_path]
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

    raise RuntimeError(f"Could not decode {path}. Details: {'; '.join(errors)}")


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

    def phoneme_token_len(self, text: str, max_tokens: int = MAX_TOKENS) -> int:
        ps = self.phonemizer.phonemize([text.strip()])
        ps = " ".join(word_tokenize(ps[0]))
        tokens = textcleaner(ps)
        return len(tokens) + 1  # leading 0 token in synthesize_sentence

    def split_long_sentence(self, sentence: str, max_tokens: int = MAX_TOKENS) -> list[str]:
        words = sentence.split()
        chunks, buf = [], []
        for w in words:
            trial = " ".join(buf + [w])
            if buf and self.phoneme_token_len(trial, max_tokens) > max_tokens:
                piece = " ".join(buf)
                if piece[-1] not in ".!?":
                    piece += "."
                chunks.append(piece)
                buf = [w]
            else:
                buf.append(w)
        if buf:
            piece = " ".join(buf)
            if piece[-1] not in ".!?":
                piece += "."
            chunks.append(piece)
        return chunks

    def chunk_by_tokens(self, text: str, max_tokens: int = MAX_TOKENS) -> list[str]:
        chunks, buf = [], ""
        for sent in split_sentences(text):
            pieces = (
                self.split_long_sentence(sent, max_tokens)
                if self.phoneme_token_len(sent, max_tokens) > max_tokens
                else [sent]
            )
            for piece in pieces:
                trial = (buf + " " + piece).strip() if buf else piece
                if buf and self.phoneme_token_len(trial, max_tokens) > max_tokens:
                    chunks.append(buf)
                    buf = piece
                else:
                    buf = trial
        if buf:
            chunks.append(buf)
        return chunks

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
        alpha: float = ALPHA,
        beta: float = BETA,
        t: float = STYLE_BLEND_T,
        diffusion_steps: int = DIFFUSION_STEPS,
        embedding_scale: float = 1.0,
        speed: float = SPEED,
        max_tokens: int = MAX_TOKENS,
    ):
        text = text.strip().replace('"', "")
        ps = self.phonemizer.phonemize([text])
        ps = " ".join(word_tokenize(ps[0])).replace("``", '"').replace("''", '"')
        tokens = textcleaner(ps)
        tokens.insert(0, 0)
        if len(tokens) > max_tokens:
            raise ValueError(f"Chunk too long ({len(tokens)} tokens): {text[:80]}...")
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
        alpha: float = ALPHA,
        beta: float = BETA,
        diffusion_steps: int = DIFFUSION_STEPS,
        embedding_scale: float = 1.0,
        speed: float = SPEED,
        max_tokens: int = MAX_TOKENS,
    ) -> np.ndarray:
        """Synthesize long text: token chunks + s_prev + crossfade (no per-sentence silence)."""
        chunks = self.chunk_by_tokens(text, max_tokens=max_tokens)
        if not chunks:
            raise ValueError("Empty utterance")

        print(f"    {len(chunks)} chunk(s), tokens ≤{max_tokens}", flush=True)
        wav_out: np.ndarray | None = None
        s_prev = None
        for i, chunk in enumerate(chunks):
            ntok = self.phoneme_token_len(chunk, max_tokens)
            print(f"    chunk [{i + 1}/{len(chunks)}] tokens={ntok}  {chunk[:72]}...", flush=True)
            w, s_prev = self.synthesize_sentence(
                chunk,
                ref_s,
                s_prev=s_prev,
                alpha=alpha,
                beta=beta,
                t=STYLE_BLEND_T,
                diffusion_steps=diffusion_steps,
                embedding_scale=embedding_scale,
                speed=speed,
                max_tokens=max_tokens,
            )
            w = trim_pulse(w)
            wav_out = w if wav_out is None else crossfade(wav_out, w)
        assert wav_out is not None
        return wav_out.astype(np.float32)


def _normalize_speaker(raw: str) -> str | None:
    s = str(raw).strip().upper()
    if s in SPEAKERS:
        return s
    if s in {"OPENAI", "AI", "ASSISTANT", "ASSISTANT_AI", "BOT", "SPEAKER_A", "A", "AGENT", "SANDRA", "FAITH"}:
        return "FAITH"
    if s in {"BIMPE", "BIMPEAI", "BIMPE_AI", "SPEAKER_B", "B", "HUMAN", "CUSTOMER", "ANDREW", "JAMES"}:
        return "JAMES"
    return None


_CURRENCY_RE = re.compile(r"\$\s*([\d,]+(?:\.\d{1,2})?)")
_DIGITS_RE = re.compile(r"\d+")


def _parse_amount(raw: str) -> float:
    return float(raw.replace(",", ""))


def _currency_to_words(raw: str) -> str:
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
    try:
        from num2words import num2words
    except ImportError as e:
        raise SystemExit("Install num2words: pip install num2words") from e
    return " ".join(num2words(int(ch), lang="en") for ch in raw)


def numbers_to_words(text: str) -> str:
    text = _CURRENCY_RE.sub(lambda m: _currency_to_words(m.group(1)), text)
    return _DIGITS_RE.sub(lambda m: _digits_to_words(m.group(0)), text)


def generate_dialogue(topic: str, turns: int, model: str = CHAT_MODEL) -> list[dict]:
    try:
        from openai import OpenAI
    except ImportError as e:
        raise SystemExit("Install openai: pip install openai\nAnd set OPENAI_API_KEY.") from e
    if not env("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY in the environment.")

    client = OpenAI()
    system = (
        "You write spoken bank customer-care dialogue scripts for a TTS demo. "
        'Return ONLY valid JSON with key "turns": a list of objects with keys '
        '"speaker" (either "FAITH" or "JAMES") and "text" (one spoken sentence). '
        "FAITH is a polite Verca Bank customer care representative handling loan inquiries. "
        "JAMES is a customer who has just called to ask about taking a loan. "
        "Cover greeting, loan purpose, eligibility basics, documents, next steps, closing. "
        "Alternate speakers. Start with FAITH. Each line under 30 words. No stage directions."
    )
    user = (
        f"Write a natural {turns}-turn phone conversation about: {topic}. "
        f"Bank: Verca Bank. Characters: Faith (agent), James (customer). "
        f"Exactly {turns} objects in the turns array."
    )
    resp = client.chat.completions.create(
        model=model,
        temperature=0.75,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
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
        cleaned.append({"speaker": speaker, "text": numbers_to_words(text)})
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


def resolve_checkpoint(path: str) -> str:
    if os.path.isfile(path):
        return path
    return find_best_or_latest("Models/BimpeTTS_ft")


def main() -> int:
    checkpoint = resolve_checkpoint(CHECKPOINT_PATH)
    for label, path in [("config", CONFIG_PATH), ("checkpoint", checkpoint)]:
        if not os.path.isfile(path):
            raise SystemExit(f"Missing {label}: {path}")

    resolved_refs: dict[str, str] = {}
    for speaker, ref_path in SPEAKER_REFS.items():
        try:
            resolved_refs[speaker] = resolve_audio_path(ref_path)
        except Exception as e:
            raise SystemExit(f"Bad reference for {speaker} ({ref_path}): {e}") from e

    print("Voice backend: StyleTTS2 (upgrade: token chunks + crossfade)", flush=True)
    for speaker, path in resolved_refs.items():
        audio = load_audio_mono_24k(path)
        print(f"  {speaker}: {path} ({len(audio) / SR:.2f}s)", flush=True)

    print("Generating dialogue text with OpenAI Chat...", flush=True)
    dialogue = generate_dialogue(TOPIC, TURNS, model=CHAT_MODEL)
    for i, turn in enumerate(dialogue, 1):
        print(f"  [{i}] {turn['speaker']}: {turn['text']}", flush=True)

    if SCRIPT_JSON_PATH is not None:
        SCRIPT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
        SCRIPT_JSON_PATH.write_text(json.dumps(dialogue, indent=2), encoding="utf-8")
        print(f"Saved script -> {SCRIPT_JSON_PATH}", flush=True)

    print("Loading StyleTTS2...", flush=True)
    tts = StyleTTSRuntime(CONFIG_PATH, checkpoint)
    styles = {speaker: tts.compute_style(path) for speaker, path in resolved_refs.items()}

    gap = np.zeros(int(SR * GAP_MS / 1000.0), dtype=np.float32)
    parts: list[np.ndarray] = []
    for i, turn in enumerate(dialogue):
        speaker = turn["speaker"]
        print(f"Synthesizing {speaker}...", flush=True)
        wav = tts.synthesize_text(turn["text"], styles[speaker])
        parts.append(wav)
        if i < len(dialogue) - 1:
            parts.append(gap)

    save_wav(OUTPUT_PATH, np.concatenate(parts))
    transcript_path = OUTPUT_PATH.with_suffix(".txt")
    transcript_path.write_text(
        "\n".join(f"{t['speaker']}: {t['text']}" for t in dialogue) + "\n",
        encoding="utf-8",
    )
    print(f"Transcript -> {transcript_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
