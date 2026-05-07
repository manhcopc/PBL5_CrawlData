from fastapi import APIRouter, HTTPException

from app.schemas.payload import DesignRequest, GenerationJobAccepted, GenerationJobStatus
from app.services.gen_service import gen_service
from app.services.generation_jobs import generation_job_manager

router = APIRouter()


@router.post("/generate-design", response_model=GenerationJobAccepted, status_code=202)
async def generate_design_endpoint(payload: DesignRequest):
    print(f"[API] Queue generation request {payload.request_id}: {payload.target_style_prompt}")
    job = await generation_job_manager.submit(payload)
    return {
        "status": "queued",
        "request_id": payload.request_id,
        "job_id": job.job_id,
    }


@router.get("/generation-jobs/{job_id}", response_model=GenerationJobStatus)
async def get_generation_job(job_id: str):
    job = await generation_job_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Generation job not found.")
    return job.as_status()


@router.post("/warmup/gen")
async def warmup_generation_service():
    if not gen_service.ensure_ready():
        raise HTTPException(
            status_code=500,
            detail=f"Gen Service is offline: {gen_service.load_error}",
        )
    return {
        "status": "ready",
        "device": gen_service.device,
    }

