# StyleTTS2 HTTP TTS server (notebook parity)

Audio matches `Demo/server_Inference_LibriTTS.ipynb` cell 25:

`chunk_by_tokens(≤500)` → `LFinference` (`s_prev`, `t=0.7`) → `trim_pulse(15ms)` → `crossfade(80ms)`.

Defaults: `alpha=0.3`, `beta=0.7`, `diffusion_steps=5`, `speed=1.0`, FP32.

## Run (GPU host)

```bash
cd /path/to/StyleTTS2-1
export CONFIG_PATH=Configs/config_bimpe_ft.yml
export CHECKPOINT_PATH=Models/best_2nd_lite.pth   # same as notebook when available
export VOICES_DIR=/path/to/voices
export STYLETTS_DEFAULT_VOICE=tara
export STYLETTS_MAX_CONCURRENT=1

uvicorn serving.tts_server:app --host 0.0.0.0 --port 8000
```

Voice stems: pure-name `.wav` under `VOICES_DIR` (skip `*_sample`). Compatible set:
`bimpe-ai-onprem-realtime-STT/services/tts/voices/`.

## API

| Method | Path | Notes |
|--------|------|--------|
| GET | `/health` | `ready`, `device`, `voices_loaded`, notebook `defaults` |
| GET | `/voices` | list + default |
| POST | `/tts` | JSON → `audio/wav` @ 24 kHz (primary) |
| POST | `/tts/stream` | Same synth as `/tts`, then raw s16le PCM of that waveform |

Body: `{text, voice?, alpha?, beta?, diffusion_steps?, embedding_scale?, speed?}`.

## Agents

```bash
export STYLETTS_URL=http://127.0.0.1:8000
export STYLETTS_ALPHA=0.3 STYLETTS_BETA=0.7 STYLETTS_DIFFUSION_STEPS=5 STYLETTS_SPEED=1.0
python agent.py start
```

One LiveKit utterance → one server `synthesize_text` call (no client micro-chunking).

## Smoke

```bash
curl -s "$STYLETTS_URL/health"
curl -s -X POST "$STYLETTS_URL/tts" -H 'Content-Type: application/json' \
  -d '{"text":"Hello from StyleTTS.","voice":"tara"}' -o /tmp/styletts_smoke.wav
```
