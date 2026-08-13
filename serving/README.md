# StyleTTS2 HTTP TTS server

Always-on FastAPI process for LiveKit agents (and curl smoke tests).

## Run (GPU host)

```bash
cd /path/to/StyleTTS2-1
export CONFIG_PATH=Configs/config_bimpe_ft.yml
export CHECKPOINT_PATH=Models/BimpeTTS_ft/best_2nd.pth
export VOICES_DIR=/path/to/voices   # tara.wav, james.wav, femi.wav, …
export STYLETTS_DEFAULT_VOICE=tara
export STYLETTS_MAX_CONCURRENT=1    # raise only after measuring VRAM; scale via replicas

uvicorn serving.tts_server:app --host 0.0.0.0 --port 8000
```

Voice wav stems must match frontend IDs (lowercase; skip `*_sample`):
`agnes`, `bayo`, `bretheny`, `eli`, `femi`, `freda`, `james`, `kara`, `sarah`, `tara`, `temi`, `tobi`.

Compatible reference set — point `VOICES_DIR` at:

`bimpe-ai-onprem-realtime-STT/services/tts/voices/`

## API

| Method | Path | Notes |
|--------|------|--------|
| GET | `/health` | `ready`, `voices_loaded`, `in_flight`, `max_concurrent` |
| GET | `/voices` | list + default |
| POST | `/tts` | JSON `{text, voice?, alpha?, beta?, diffusion_steps?, speed?}` → `audio/wav` @ 24 kHz |

Synthesis uses phoneme-token chunks (≤500), style continuity (`s_prev`), trim + crossfade between chunks.

## Concurrency

One process holds one GPU model. Default `STYLETTS_MAX_CONCURRENT=1` queues overlapping requests. For more LiveKit rooms, run **horizontal replicas** (separate GPUs / processes) behind a load balancer — do not set concurrent=10 on a single card.

## Agents

Point bimpe-agents at this service:

```bash
export STYLETTS_URL=http://<gpu-host>:8000
python agent.py dev
```

See also bimpe-agents `STYLETTS.md`.

## Smoke

```bash
curl -s "$STYLETTS_URL/health"
curl -s "$STYLETTS_URL/voices"
curl -s -X POST "$STYLETTS_URL/tts" -H 'Content-Type: application/json' \
  -d '{"text":"Hello from StyleTTS.","voice":"tara"}' -o /tmp/styletts_smoke.wav
```

Then one LiveKit room from the frontend; then 2–4 rooms and watch `in_flight` vs OOM. Keep `STYLETTS_MAX_CONCURRENT` at 1–2 on a single GPU unless you measured VRAM.