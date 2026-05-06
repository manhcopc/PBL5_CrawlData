# TrendEngine AI Core

FastAPI microservice for a zero-shot fashion pipeline:

- PhoBERT sentiment inference ranks Vietnamese e-commerce products.
- Stable Diffusion 1.5 + ControlNet Canny maps trend prompts onto a base garment shape.
- A single in-process generation worker protects Colab Tesla T4 VRAM.

## Colab Runtime

```bash
pip install -r requirements.txt
export PUBLIC_BASE_URL="https://<your-ngrok-domain>"
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Useful endpoints:

- `GET /health`: process health and generation queue depth.
- `GET /ready`: lazy-loaded model readiness and load errors.
- `POST /api/v1/warmup/gen`: load Stable Diffusion + ControlNet before the first generation job.
- `POST /api/v1/analyze-trend`: PhoBERT sentiment and training-free trend scoring.
- `POST /api/v1/generate-design`: enqueue a generation job and return `job_id`.
- `GET /api/v1/generation-jobs/{job_id}`: poll queued/running/succeeded/failed generation state.
