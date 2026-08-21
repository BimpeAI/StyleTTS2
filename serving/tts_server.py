"""
FastAPI TTS server — audio path matches Demo/server_Inference_LibriTTS.ipynb cell 25.

Long-form: chunk_by_tokens(≤500) → LFinference (s_prev, t=0.7) → trim_pulse(15ms)
→ crossfade(80ms). Defaults: alpha=0.3, beta=0.7, diffusion_steps=5, speed=1.0, FP32.

Run from StyleTTS2 repo root:
  export CONFIG_PATH=Configs/config_bimpe_ft.yml
  export CHECKPOINT_PATH=Models/best_2nd_lite.pth
  export VOICES_DIR=/path/to/voices
  export STYLETTS_DEFAULT_VOICE=tara
  uvicorn serving.tts_server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import glob
import io
import os
import re
import sys
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import torch
import torchaudio
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from nltk.tokenize import word_tokenize
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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

# Notebook cell-25 / LFinference defaults
DEFAULT_ALPHA = 0.3
DEFAULT_BETA = 0.7
DEFAULT_DIFFUSION_STEPS = 5
DEFAULT_EMBEDDING_SCALE = 1.0
DEFAULT_SPEED = 1.0
DEFAULT_STYLE_BLEND_T = 0.7
MAX_TOKENS = 500
TRIM_TAIL_MS = 15
TRIM_HEAD_MS = 4
CROSSFADE_MS = 80

to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300
)
textcleaner = TextCleaner()


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to synthesize")
    voice: Optional[str] = Field(
        None, description="Voice id = pure wav stem (e.g. tara)."
    )
    alpha: float = DEFAULT_ALPHA
    beta: float = DEFAULT_BETA
    diffusion_steps: int = DEFAULT_DIFFUSION_STEPS
    embedding_scale: float = DEFAULT_EMBEDDING_SCALE
    speed: float = Field(DEFAULT_SPEED, gt=0)


class AppState:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config = None
        self.model = None
        self.model_params = None
        self.sampler = None
        self.phonemizer = None
        self.ref_s = None
        self.styles: dict[str, torch.Tensor] = {}
        self.voices_dir: Optional[str] = None
        self.default_voice: str = "tara"
        self.checkpoint_path = None
        self.ref_wav = None
        self.synth_semaphore: Optional[asyncio.Semaphore] = None
        self.max_concurrent: int = 1
        self.in_flight: int = 0


state = AppState()


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    return os.environ.get(name, default)


def _is_sample_voice(stem: str) -> bool:
    return stem.endswith("_sample")


def find_latest_checkpoint(log_dir: str) -> str:
    for name in ("best_2nd_lite.pth", "best_2nd.pth"):
        path = os.path.join(log_dir, name)
        if os.path.isfile(path):
            return path
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
            if next(iter(state_dict)).startswith("module."):
                for k, v in state_dict.items():
                    new_state_dict[k[7:] if k.startswith("module.") else k] = v
            else:
                new_state_dict = state_dict
            model[key].load_state_dict(new_state_dict, strict=False)
    _ = [model[key].eval() for key in model]
    return params_whole.get("epoch"), params_whole.get("val_loss")


def compute_style(path: str) -> torch.Tensor:
    """Match notebook compute_style (librosa 24k + trim + style/predictor encoders)."""
    wave, sr = librosa.load(path, sr=24000)
    audio, _ = librosa.effects.trim(wave, top_db=30)
    if sr != 24000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=24000)
    mel_tensor = preprocess(audio.astype(np.float32)).to(state.device)
    with torch.no_grad():
        ref_s = state.model.style_encoder(mel_tensor.unsqueeze(1))
        ref_p = state.model.predictor_encoder(mel_tensor.unsqueeze(1))
    return torch.cat([ref_s, ref_p], dim=1)


def list_voice_wavs(voices_dir: str) -> list[tuple[str, Path]]:
    root = Path(voices_dir)
    if not root.is_dir():
        return []
    out: list[tuple[str, Path]] = []
    for path in sorted(root.glob("*.wav")):
        if _is_sample_voice(path.stem):
            continue
        out.append((path.stem, path))
    return out


def resolve_ref_s(voice: Optional[str]) -> tuple[str, torch.Tensor]:
    default = state.default_voice
    name = (voice or default or "").strip() or default
    if _is_sample_voice(name):
        raise ValueError(
            f"voice '{name}' is a *_sample name; available: {sorted(state.styles)}"
        )
    if name in state.styles:
        return name, state.styles[name]
    if state.ref_s is not None and (not name or name == default):
        return default or "default", state.ref_s
    raise ValueError(f"Unknown voice '{name}'. Available: {sorted(state.styles)}")


def split_sentences(text: str) -> list[str]:
    text = text.strip().replace('"', "")
    parts = re.split(r"(?<=[.!?])\s+", text)
    out: list[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if p[-1] not in ".!?":
            p += "."
        out.append(p)
    return out


def phoneme_token_len(text: str) -> int:
    ps = state.phonemizer.phonemize([text.strip()])
    ps = " ".join(word_tokenize(ps[0]))
    tokens = textcleaner(ps)
    return len(tokens) + 1


def split_long_sentence(sentence: str, max_tokens: int = MAX_TOKENS) -> list[str]:
    words = sentence.split()
    chunks, buf = [], []
    for w in words:
        trial = " ".join(buf + [w])
        if buf and phoneme_token_len(trial) > max_tokens:
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


def chunk_by_tokens(text: str, max_tokens: int = MAX_TOKENS) -> list[str]:
    chunks, buf = [], ""
    for sent in split_sentences(text):
        pieces = (
            split_long_sentence(sent, max_tokens)
            if phoneme_token_len(sent) > max_tokens
            else [sent]
        )
        for piece in pieces:
            trial = (buf + " " + piece).strip() if buf else piece
            if buf and phoneme_token_len(trial) > max_tokens:
                chunks.append(buf)
                buf = piece
            else:
                buf = trial
    if buf:
        chunks.append(buf)
    return chunks


def trim_pulse(
    w: np.ndarray,
    tail_ms: int = TRIM_TAIL_MS,
    head_ms: int = TRIM_HEAD_MS,
    sr: int = SR,
) -> np.ndarray:
    """Notebook trim_pulse — drop end-click and a bit of leading noise."""
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
    """Notebook cosine overlap-add (cell 25 uses fade_ms=80)."""
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    n = min(int(sr * fade_ms / 1000), len(a) // 4, len(b) // 4)
    if n < 8:
        return np.concatenate([a, b])
    fade_out = 0.5 * (1.0 + np.cos(np.linspace(0, np.pi, n, dtype=np.float32)))
    fade_in = fade_out[::-1]
    head = a[:-n]
    overlap = a[-n:] * fade_out + b[:n] * fade_in
    return np.concatenate([head, overlap, b[n:]])


def LFinference(
    text: str,
    s_prev,
    ref_s: torch.Tensor,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    t: float = DEFAULT_STYLE_BLEND_T,
    diffusion_steps: int = DEFAULT_DIFFUSION_STEPS,
    embedding_scale: float = DEFAULT_EMBEDDING_SCALE,
    speed: float = DEFAULT_SPEED,
):
    """Notebook LFinference — returns (wav[..., :-100], s_pred)."""
    text = text.strip()
    ps = state.phonemizer.phonemize([text])
    ps = word_tokenize(ps[0])
    ps = " ".join(ps)
    ps = ps.replace("``", '"').replace("''", '"')

    tokens = textcleaner(ps)
    tokens.insert(0, 0)
    tokens = torch.LongTensor(tokens).to(state.device).unsqueeze(0)

    model = state.model
    model_params = state.model_params

    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(state.device)
        text_mask = length_to_mask(input_lengths).to(state.device)

        t_en = model.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

        s_pred = state.sampler(
            noise=torch.randn((1, 256)).unsqueeze(1).to(state.device),
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
        duration = model.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1)
        pred_dur = torch.round(duration.squeeze() / speed).clamp(min=1)

        pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().item()))
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame : c_frame + int(pred_dur[i].item())] = 1
            c_frame += int(pred_dur[i].item())

        en = d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(state.device)
        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(en)
            asr_new[:, :, 0] = en[:, :, 0]
            asr_new[:, :, 1:] = en[:, :, 0:-1]
            en = asr_new

        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)

        asr = t_en @ pred_aln_trg.unsqueeze(0).to(state.device)
        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(asr)
            asr_new[:, :, 0] = asr[:, :, 0]
            asr_new[:, :, 1:] = asr[:, :, 0:-1]
            asr = asr_new

        out = model.decoder(asr, F0_pred, N_pred, ref.squeeze().unsqueeze(0))

    return out.squeeze().cpu().numpy()[..., :-100], s_pred


def synthesize_text(
    text: str,
    ref_s: torch.Tensor,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    diffusion_steps: int = DEFAULT_DIFFUSION_STEPS,
    embedding_scale: float = DEFAULT_EMBEDDING_SCALE,
    speed: float = DEFAULT_SPEED,
) -> np.ndarray:
    """Notebook cell-25 loop: LFinference + trim_pulse(15) + crossfade(80)."""
    chunks = chunk_by_tokens(text, max_tokens=MAX_TOKENS)
    if not chunks:
        raise ValueError("No sentences found in text")

    wav_out: Optional[np.ndarray] = None
    s_prev = None
    for chunk in chunks:
        wav, s_prev = LFinference(
            chunk,
            s_prev,
            ref_s,
            alpha=alpha,
            beta=beta,
            t=DEFAULT_STYLE_BLEND_T,
            diffusion_steps=diffusion_steps,
            embedding_scale=embedding_scale,
            speed=speed,
        )
        wav = trim_pulse(wav, tail_ms=TRIM_TAIL_MS, head_ms=TRIM_HEAD_MS)
        wav_out = wav if wav_out is None else crossfade(wav_out, wav, fade_ms=CROSSFADE_MS)
    assert wav_out is not None
    return wav_out


def audio_to_wav_bytes(audio: np.ndarray, sr: int = SR) -> bytes:
    audio = np.asarray(audio, dtype=np.float32)
    peak = np.max(np.abs(audio)) + 1e-8
    if peak > 1.0:
        audio = audio / peak
    buf = io.BytesIO()
    torchaudio.save(buf, torch.from_numpy(audio).unsqueeze(0), sr, format="wav")
    return buf.getvalue()


def audio_to_pcm16_bytes(audio: np.ndarray) -> bytes:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    peak = float(np.max(np.abs(audio))) + 1e-8
    if peak > 1.0:
        audio = audio / peak
    return (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


def _load_voice_styles(voices_dir: Optional[str]) -> None:
    state.styles = {}
    state.voices_dir = voices_dir
    if not voices_dir:
        return
    pairs = list_voice_wavs(voices_dir)
    if not pairs:
        print(f"VOICES_DIR={voices_dir}: no pure-name .wav files", flush=True)
        return
    for stem, path in pairs:
        try:
            state.styles[stem] = compute_style(str(path))
            print(f"  voice loaded: {stem} <- {path}", flush=True)
        except Exception as e:
            print(f"  voice FAILED: {stem} ({path}): {e}", flush=True)


def load_runtime():
    config_path = _env("CONFIG_PATH", "Configs/config_bimpe_ft.yml")
    voices_dir = _env("VOICES_DIR") or _env("STYLETTS_VOICES_DIR")
    default_voice = (_env("STYLETTS_DEFAULT_VOICE", "tara") or "tara").strip()
    ref_wav = _env("REF_WAV")
    log_dir_default = "Models"

    if state.device != "cuda":
        print(
            "WARNING: StyleTTS on CPU — same math as notebook, much slower.",
            flush=True,
        )

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"CONFIG_PATH not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    ckpt = _env("CHECKPOINT_PATH")
    if not ckpt:
        # Prefer notebook lite checkpoint under Models/ or config log_dir
        for candidate in (
            "Models/best_2nd_lite.pth",
            os.path.join(config.get("log_dir", log_dir_default), "best_2nd_lite.pth"),
        ):
            if os.path.isfile(candidate):
                ckpt = candidate
                break
        if not ckpt:
            ckpt = find_latest_checkpoint(config.get("log_dir", log_dir_default))
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"CHECKPOINT_PATH not found: {ckpt}")

    import nltk
    import phonemizer
    from Utils.PLBERT.util import load_plbert

    for pkg in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)

    device = state.device
    text_aligner = load_ASR_models(config.get("ASR_path"), config.get("ASR_config"))
    pitch_extractor = load_F0_models(config.get("F0_path"))
    plbert = load_plbert(config.get("PLBERT_dir"))

    model_params = recursive_munch(config["model_params"])
    model = build_model(model_params, text_aligner, pitch_extractor, plbert)
    _ = [model[key].eval() for key in model]
    _ = [model[key].to(device) for key in model]

    epoch, val_loss = load_checkpoint_weights(model, ckpt)
    sampler = DiffusionSampler(
        model.diffusion.diffusion,
        sampler=ADPM2Sampler(),
        sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
        clamp=False,
    )

    state.config = config
    state.model = model
    state.model_params = model_params
    state.sampler = sampler
    state.phonemizer = phonemizer.backend.EspeakBackend(
        language="en-us", preserve_punctuation=True, with_stress=True
    )
    state.checkpoint_path = ckpt
    state.default_voice = default_voice

    _load_voice_styles(voices_dir)

    if default_voice in state.styles:
        state.ref_s = state.styles[default_voice]
        state.ref_wav = (
            str(Path(voices_dir) / f"{default_voice}.wav") if voices_dir else default_voice
        )
    elif ref_wav and os.path.isfile(ref_wav):
        state.ref_s = compute_style(ref_wav)
        state.ref_wav = ref_wav
        if default_voice and default_voice not in state.styles and not _is_sample_voice(
            default_voice
        ):
            state.styles[default_voice] = state.ref_s
    elif state.styles:
        first = sorted(state.styles)[0]
        state.default_voice = first
        state.ref_s = state.styles[first]
        state.ref_wav = str(Path(voices_dir) / f"{first}.wav") if voices_dir else first
        print(f"Default voice '{default_voice}' missing; using '{first}'", flush=True)
    else:
        raise FileNotFoundError(
            "No voice styles loaded. Set VOICES_DIR to pure-name .wav files "
            f"and/or REF_WAV. Default was '{default_voice}'."
        )

    max_c = max(1, int(_env("STYLETTS_MAX_CONCURRENT", "1") or "1"))
    state.max_concurrent = max_c
    state.in_flight = 0
    state.synth_semaphore = asyncio.Semaphore(max_c)

    print(
        f"TTS ready (notebook-parity) | device={device} | ckpt={ckpt} | epoch={epoch} "
        f"| val_loss={val_loss} | default_voice={state.default_voice} "
        f"| voices={sorted(state.styles)} | max_concurrent={max_c} "
        f"| alpha={DEFAULT_ALPHA} beta={DEFAULT_BETA} steps={DEFAULT_DIFFUSION_STEPS} "
        f"| speed={DEFAULT_SPEED} | trim_tail_ms={TRIM_TAIL_MS} crossfade_ms={CROSSFADE_MS}",
        flush=True,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_runtime()
    yield


app = FastAPI(title="StyleTTS2 Server (LibriTTS notebook parity)", lifespan=lifespan)


@app.get("/health")
def health():
    ready = state.model is not None and state.ref_s is not None
    return {
        "status": "ok" if ready else "loading",
        "ready": ready,
        "device": state.device,
        "parity": "server_Inference_LibriTTS.ipynb",
        "checkpoint": state.checkpoint_path,
        "ref_wav": state.ref_wav,
        "default_voice": state.default_voice,
        "voices_loaded": sorted(state.styles.keys()),
        "voices_dir": state.voices_dir,
        "in_flight": state.in_flight,
        "max_concurrent": state.max_concurrent,
        "defaults": {
            "alpha": DEFAULT_ALPHA,
            "beta": DEFAULT_BETA,
            "diffusion_steps": DEFAULT_DIFFUSION_STEPS,
            "speed": DEFAULT_SPEED,
            "trim_tail_ms": TRIM_TAIL_MS,
            "crossfade_ms": CROSSFADE_MS,
        },
    }


@app.get("/voices")
def voices():
    return {
        "voices": sorted(state.styles.keys()),
        "default": state.default_voice,
    }


async def _run_synth(req: TTSRequest, ref_s: torch.Tensor) -> np.ndarray:
    sem = state.synth_semaphore or asyncio.Semaphore(1)

    def _work():
        return synthesize_text(
            req.text,
            ref_s,
            alpha=req.alpha,
            beta=req.beta,
            diffusion_steps=req.diffusion_steps,
            embedding_scale=req.embedding_scale,
            speed=req.speed,
        )

    async with sem:
        state.in_flight += 1
        try:
            return await asyncio.to_thread(_work)
        finally:
            state.in_flight = max(0, state.in_flight - 1)


@app.post("/tts")
async def tts(req: TTSRequest):
    if state.model is None or state.ref_s is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        voice_name, ref_s = resolve_ref_s(req.voice)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    try:
        audio = await _run_synth(req, ref_s)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS failed: {e}") from e
    return Response(
        content=audio_to_wav_bytes(audio),
        media_type="audio/wav",
        headers={"X-StyleTTS-Voice": voice_name},
    )


@app.post("/tts/stream")
async def tts_stream(req: TTSRequest):
    """Same notebook synth as /tts, then stream s16le PCM of that full waveform."""
    if state.model is None or state.ref_s is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        voice_name, ref_s = resolve_ref_s(req.voice)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    try:
        audio = await _run_synth(req, ref_s)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS failed: {e}") from e

    pcm = audio_to_pcm16_bytes(audio)
    frame = (SR // 20) * 2  # 50 ms

    async def _gen():
        for i in range(0, len(pcm), frame):
            yield pcm[i : i + frame]

    return StreamingResponse(
        _gen(),
        media_type="application/octet-stream",
        headers={
            "X-StyleTTS-Voice": voice_name,
            "X-StyleTTS-Sample-Rate": str(SR),
            "X-StyleTTS-Encoding": "s16le",
            "X-StyleTTS-Channels": "1",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "serving.tts_server:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
        reload=False,
    )
