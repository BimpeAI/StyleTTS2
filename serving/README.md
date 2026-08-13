# StyleTTS2 HTTP TTS server

Always-on FastAPI process for LiveKit agents. Optimized for **TTFA &lt; 200ms** on short first phrases (warm CUDA, non-queued).

## Run (GPU host)

```bash
cd /path/to/StyleTTS2-1
export CONFIG_PATH=Configs/config_bimpe_ft.yml
export CHECKPOINT_PATH=Models/BimpeTTS_ft/best_2nd.pth
export VOICES_DIR=/path/to/voices
export STYLETTS_DEFAULT_VOICE=tara
export STYLETTS_MAX_CONCURRENT=1
export STYLETTS_REQUIRE_CUDA=1
export STYLETTS_FP16=1
# export STYLETTS_TORCH_COMPILE=1   # optional, after validating quality

uvicorn serving.tts_server:app --host 0.0.0.0 --port 8000
```

Voice stems (lowercase; skip `*_sample`):  
`agnes`, `bayo`, `bretheny`, `eli`, `femi`, `freda`, `james`, `kara`, `sarah`, `tara`, `temi`, `tobi`  
— point `VOICES_DIR` at `bimpe-ai-onprem-realtime-STT/services/tts/voices/`.

Defaults: `diffusion_steps=2`, `speed=1.4`, `alpha=0.0`, `beta=0.2`. Startup runs a short warmup synth.

## API

| Method | Path | Notes |
|--------|------|--------|
| GET | `/health` | `ready`, `device`, `fp16`, `warmed_up`, `voices_loaded`, `in_flight`, `synth_ms_p50` / `p95` |
| GET | `/voices` | list + default |
| POST | `/tts` | JSON → `audio/wav` @ 24 kHz (smoke / fallback) |
| POST | `/tts/stream` | JSON → chunked raw **s16le** mono 24 kHz (agents; low TTFA) |

Body: `{text, voice?, alpha?, beta?, diffusion_steps?, speed?}`.

## Latency SLA

- **In scope:** first PCM bytes for a short phrase (`"Hi there."`) with `device=cuda`, warmed model, `queue_wait≈0`.
- **Out of scope:** finishing long paragraphs in &lt;200ms; multi-room queue wait.

## Concurrency

Default `STYLETTS_MAX_CONCURRENT=1`. Scale with **replicas**, not a high concurrent setting on one GPU.

## Agents

```bash
export STYLETTS_URL=http://127.0.0.1:8000   # same host preferred
python agent.py start
```

See bimpe-agents `STYLETTS.md`.

## Smoke

```bash
curl -s "$STYLETTS_URL/health"
curl -s -X POST "$STYLETTS_URL/tts" -H 'Content-Type: application/json' \
  -d '{"text":"Hi there.","voice":"tara"}' -o /tmp/styletts_smoke.wav
curl -s -X POST "$STYLETTS_URL/tts/stream" -H 'Content-Type: application/json' \
  -d '{"text":"Hi there.","voice":"tara"}' -o /tmp/styletts.pcm
# Server log should show ttfa_ms=… ; /health synth_ms_p50 after a few calls
```
