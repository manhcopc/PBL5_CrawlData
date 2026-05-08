from fastapi import APIRouter, HTTPException

from app.schemas.payload import DesignRequest, GenerationJobAccepted, GenerationJobStatus
from app.services.gen_service import gen_service
from app.services.generation_jobs import GenerationQueueFull, generation_job_manager
from app.services.prompt_engine import build_dynamic_prompt, extract_prompt_keywords

router = APIRouter()


@router.post("/generate-design", response_model=GenerationJobAccepted, status_code=202)
async def generate_design_endpoint(payload: DesignRequest):
    prompt_keywords = extract_prompt_keywords(payload.target_style_prompt)
    base_style = prompt_keywords[0] if prompt_keywords else payload.target_style_prompt
    keywords = prompt_keywords[1:]
    season_context = f"{payload.target_season} {payload.target_weather}".strip()
    final_prompt = build_dynamic_prompt(
        base_style=base_style,
        keywords=keywords,
        season=season_context,
        audience=payload.target_audience,
    )
    enriched_payload = payload.model_copy(update={"target_style_prompt": final_prompt})

    print(f"[API] Queue generation request {payload.request_id}: {enriched_payload.target_style_prompt}")
    try:
        job = await generation_job_manager.submit(enriched_payload)
    except GenerationQueueFull as exc:
        raise HTTPException(status_code=429, detail="Generation queue is full. Try again later.") from exc
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
