import asyncio
import contextlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from app.core.config import settings
from app.schemas.payload import DesignRequest, GeneratedDesign, GenerationJobStatus
from app.services.gen_service import gen_service


@dataclass
class GenerationJob:
    job_id: str
    request: DesignRequest
    status: str = "queued"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    generated_designs: List[GeneratedDesign] = field(default_factory=list)
    error: Optional[str] = None

    def as_status(self) -> GenerationJobStatus:
        return GenerationJobStatus(
            job_id=self.job_id,
            request_id=self.request.request_id,
            status=self.status,
            created_at=self.created_at,
            updated_at=self.updated_at,
            started_at=self.started_at,
            finished_at=self.finished_at,
            generated_designs=self.generated_designs,
            error=self.error,
        )


class GenerationJobManager:
    def __init__(self) -> None:
        self._jobs: Dict[str, GenerationJob] = {}
        self._queue: Optional[asyncio.Queue[str]] = None
        self._workers: List[asyncio.Task] = []
        self._lock: Optional[asyncio.Lock] = None

    def start(self) -> None:
        if self._queue is None:
            self._queue = asyncio.Queue()
        if self._lock is None:
            self._lock = asyncio.Lock()
        if self._workers:
            return

        concurrency = max(1, settings.GENERATION_QUEUE_CONCURRENCY)
        for index in range(concurrency):
            self._workers.append(asyncio.create_task(self._worker(index + 1)))
        print(f"[Gen Queue] Started {concurrency} generation worker(s).")

    async def stop(self) -> None:
        for worker in self._workers:
            worker.cancel()
        for worker in self._workers:
            with contextlib.suppress(asyncio.CancelledError):
                await worker
        self._workers.clear()

    async def submit(self, request: DesignRequest) -> GenerationJob:
        self.start()
        await self.cleanup_expired()
        job = GenerationJob(job_id=uuid.uuid4().hex, request=request)
        assert self._lock is not None
        assert self._queue is not None
        async with self._lock:
            self._jobs[job.job_id] = job
        await self._queue.put(job.job_id)
        return job

    async def get(self, job_id: str) -> Optional[GenerationJob]:
        if self._lock is None:
            return self._jobs.get(job_id)
        async with self._lock:
            return self._jobs.get(job_id)

    def stats(self) -> Dict[str, int]:
        counts = {"queued": 0, "running": 0, "succeeded": 0, "failed": 0}
        for job in self._jobs.values():
            counts[job.status] = counts.get(job.status, 0) + 1
        counts["queue_depth"] = self._queue.qsize() if self._queue is not None else 0
        return counts

    async def cleanup_expired(self) -> int:
        if self._lock is None:
            self._lock = asyncio.Lock()
        cutoff = datetime.now(timezone.utc) - timedelta(hours=settings.GENERATION_JOB_TTL_HOURS)
        removed = 0
        async with self._lock:
            for job_id, job in list(self._jobs.items()):
                if job.updated_at < cutoff and job.status in {"succeeded", "failed"}:
                    del self._jobs[job_id]
                    removed += 1
        return removed

    async def _worker(self, worker_id: int) -> None:
        assert self._queue is not None
        while True:
            job_id = await self._queue.get()
            try:
                await self._run_job(job_id, worker_id)
            finally:
                self._queue.task_done()

    async def _run_job(self, job_id: str, worker_id: int) -> None:
        job = await self.get(job_id)
        if job is None:
            return

        now = datetime.now(timezone.utc)
        job.status = "running"
        job.started_at = now
        job.updated_at = now
        print(f"[Gen Queue] Worker {worker_id} started job {job_id}.")

        try:
            results = await asyncio.to_thread(
                gen_service.generate_design,
                base_image_url=str(job.request.base_image_url),
                target_prompt=job.request.target_style_prompt,
                num_images=job.request.num_images,
                seed=job.request.seed,
                canny_low_threshold=job.request.canny_low_threshold,
                canny_high_threshold=job.request.canny_high_threshold,
            )
            job.generated_designs = [GeneratedDesign(**item) for item in results]
            job.status = "succeeded"
            print(f"[Gen Queue] Job {job_id} succeeded.")
        except Exception as exc:
            job.status = "failed"
            job.error = str(exc)
            print(f"[Gen Queue] Job {job_id} failed: {exc}")
        finally:
            finished = datetime.now(timezone.utc)
            job.finished_at = finished
            job.updated_at = finished


generation_job_manager = GenerationJobManager()

