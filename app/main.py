from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api import gen_routes, nlp_routes
from app.core.config import settings
from app.services.gen_service import gen_service
from app.services.generation_jobs import generation_job_manager
from app.services.nlp_service import nlp_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    generation_job_manager.start()
    gen_service.cleanup_outputs()
    yield
    await generation_job_manager.stop()


app = FastAPI(
    title="TrendEngine AI Core",
    description="Microservices AI: NLP (PhoBERT) & Computer Vision (Stable Diffusion + ControlNet)",
    version=settings.VERSION,
    lifespan=lifespan,
)

app.mount("/outputs", StaticFiles(directory=settings.OUTPUT_DIR), name="outputs")
app.include_router(nlp_routes.router, prefix="/api/v1", tags=["NLP Sentiment Analysis"])
app.include_router(gen_routes.router, prefix="/api/v1", tags=["Generative Design"])


@app.get("/health")
async def health_check():
    return {
        "status": "running",
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "public_base_url": settings.PUBLIC_BASE_URL,
        "generation_queue": generation_job_manager.stats(),
    }


@app.get("/ready")
async def readiness_check():
    nlp_ready = nlp_service.is_ready
    gen_ready = gen_service.is_ready
    return {
        "ready": nlp_ready and gen_ready,
        "services": {
            "phobert": {
                "ready": nlp_ready,
                "device": nlp_service.device,
                "model_path": settings.PHOBERT_PATH,
                "load_error": nlp_service.load_error,
            },
            "generation": {
                "ready": gen_ready,
                "device": gen_service.device,
                "load_error": gen_service.load_error,
                "queue": generation_job_manager.stats(),
            },
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)

