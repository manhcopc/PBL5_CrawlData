import contextlib
import gc
import os
import time
import uuid
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from app.core.config import settings
from app.services.gen_image_processing import ImagePreprocessor, SafeImageDownloadError, preprocess_canny_edge
from app.services.gen_prompt_engine import PromptEngineer
from app.utils.metrics_logger import log_metric


class GenerativeService:
    """Singleton service for Stable Diffusion + ControlNet image generation."""

    _instance: Optional["GenerativeService"] = None

    def __new__(cls) -> "GenerativeService":
        """Create or reuse the singleton generation service instance."""
        if cls._instance is None:
            cls._instance = super(GenerativeService, cls).__new__(cls)
            cls._instance._boot()
        return cls._instance

    def _boot(self) -> None:
        """Initialize lightweight service state without loading AI weights."""
        self.pipe: Optional[Any] = None
        self.is_ready = False
        self.load_error: Optional[str] = None
        self.device = "unknown"
        self._load_lock = Lock()
        self.output_dir = settings.OUTPUT_DIR
        self.image_preprocessor = ImagePreprocessor()
        self.prompt_engineer = PromptEngineer()
        os.makedirs(self.output_dir, exist_ok=True)
        self.cleanup_outputs()

    def ensure_ready(self) -> bool:
        """Load the diffusion pipeline once and report whether it is usable."""
        if self.is_ready:
            return True

        with self._load_lock:
            if self.is_ready:
                return True

            print("[Gen Service] Lazy-loading Stable Diffusion + ControlNet...")
            try:
                self.pipe = self._setup_pipeline()
                self.is_ready = True
                self.load_error = None
                print("[Gen Service] Stable Diffusion + ControlNet ready.")
                return True
            except Exception as exc:
                self.is_ready = False
                self.load_error = str(exc)
                print(f"[Gen Service] Load failed: {exc}")
                return False

    def _setup_pipeline(self) -> Any:
        """Create and configure the Stable Diffusion ControlNet pipeline."""
        import torch
        from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Gen Service] Device: {self.device}")

        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-canny",
            torch_dtype=torch.float16,
        )
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            torch_dtype=torch.float16,
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()
        pipe.enable_xformers_memory_efficient_attention()
        return pipe

    def _validate_public_url(self, url: str) -> None:
        """Preserve URL validation helper through the image preprocessor."""
        self.image_preprocessor.validate_public_url(url)

    def download_image(self, url: str) -> Image.Image:
        """Download, verify, and decode a public remote image as RGB."""
        return self.image_preprocessor.download_image(url)

    def resize_with_padding(
        self,
        image: Image.Image,
        size: int = settings.GENERATION_SIZE,
    ) -> Image.Image:
        """Resize an image into a square RGB canvas while preserving aspect ratio."""
        return self.image_preprocessor.resize_with_padding(image, size=size)

    def build_prompt(self, target_prompt: str) -> Dict[str, str]:
        """Build prompts for the Stable Diffusion ControlNet pipeline."""
        return self.prompt_engineer.build_prompt(target_prompt)

    def save_canny_map(self, canny_image: Image.Image) -> str:
        """Persist the latest Canny conditioning map to the project root."""
        debug_path = os.path.join(settings.ROOT_DIR, "debug_canny_latest.png")
        canny_image.save(debug_path)
        print(f"[Gen Service] Saved Canny debug map: {debug_path}")
        return debug_path

    def _resolve_canny_thresholds(
        self,
        low_threshold: Optional[int],
        high_threshold: Optional[int],
    ) -> Tuple[int, int]:
        """Resolve request-level Canny thresholds against configured defaults."""
        low = settings.CANNY_LOW_THRESHOLD if low_threshold is None else low_threshold
        high = settings.CANNY_HIGH_THRESHOLD if high_threshold is None else high_threshold
        return low, high

    def _preprocess_image(
        self,
        base_image_url: str,
        canny_low_threshold: Optional[int],
        canny_high_threshold: Optional[int],
    ) -> Image.Image:
        """Download, resize, edge-detect, and persist the ControlNet input image."""
        print(f"[Gen Service] Downloading base image: {base_image_url}")
        base_image = self.resize_with_padding(self.download_image(base_image_url))
        low_threshold, high_threshold = self._resolve_canny_thresholds(
            canny_low_threshold,
            canny_high_threshold,
        )
        canny_image = preprocess_canny_edge(
            base_image,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
        )
        self.save_canny_map(canny_image)
        return canny_image

    def _public_output_url(self, filename: str) -> str:
        """Build the public URL for a generated output file."""
        return f"{settings.PUBLIC_BASE_URL.rstrip('/')}/outputs/{filename}"

    def _make_generator(self, seed: int) -> Any:
        """Create a deterministic Torch random generator on the active device."""
        import torch

        if self.device == "cuda":
            return torch.Generator(device="cuda").manual_seed(seed)
        return torch.Generator(device="cpu").manual_seed(seed)

    def _make_image_seed(self, seed: Optional[int], index: int) -> int:
        """Create a per-image seed from a base seed or OS randomness."""
        if seed is not None:
            return seed + index
        return int.from_bytes(os.urandom(4), "big")

    def _autocast_context(self) -> Any:
        """Return CUDA autocast for GPU inference, otherwise a no-op context."""
        import torch

        if self.device == "cuda":
            return torch.autocast("cuda")
        return contextlib.nullcontext()

    def _render_image(
        self,
        prompts: Dict[str, str],
        canny_image: Image.Image,
        generator: Any,
    ) -> Image.Image:
        """Run one Stable Diffusion ControlNet inference request."""
        if self.pipe is None:
            raise RuntimeError(f"Stable Diffusion module is offline: {self.load_error}")

        import torch

        with torch.inference_mode(), self._autocast_context():
            result = self.pipe(
                prompt=prompts["prompt"],
                negative_prompt=prompts["negative_prompt"],
                image=canny_image,
                controlnet_conditioning_scale=settings.CONTROLNET_CONDITIONING_SCALE,
                num_inference_steps=settings.GENERATION_STEPS,
                guidance_scale=settings.GENERATION_GUIDANCE_SCALE,
                generator=generator,
            )
        return result.images[0]

    def _save_generated_image(self, image: Image.Image, image_seed: int) -> Dict[str, Any]:
        """Save a generated image and return its API response metadata."""
        filename = f"design_{uuid.uuid4().hex[:8]}.png"
        local_path = os.path.join(self.output_dir, filename)
        image.save(local_path)
        print(f"[Gen Service] Saved: {local_path}")
        return {
            "url": self._public_output_url(filename),
            "seed": image_seed,
        }

    def _clear_generation_memory(self) -> None:
        """Release Python and CUDA memory after each generated image."""
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _reset_gpu_peak_memory(self) -> None:
        """Reset CUDA peak memory metrics when running on GPU."""
        if self.device != "cuda":
            return

        import torch

        torch.cuda.reset_peak_memory_stats()

    def _current_peak_vram_gb(self) -> float:
        """Read peak CUDA VRAM usage in GiB for metrics logging."""
        if self.device != "cuda":
            return 0.0

        import torch

        return torch.cuda.max_memory_allocated() / (1024**3)

    def _record_generation_metrics(self, start_time: float, requested_images: int) -> None:
        """Log generation latency and peak VRAM usage."""
        latency = time.time() - start_time
        peak_vram = self._current_peak_vram_gb()
        log_metric(
            api_name="generate_design",
            latency_sec=latency,
            vram_gb=peak_vram,
            extra_info=f"num_images:{requested_images}",
        )
        print(f"[Metrics] GenAI finished in {latency:.2f}s | Peak VRAM: {peak_vram:.2f} GB")

    def generate_design(
        self,
        base_image_url: str,
        target_prompt: str,
        num_images: int = 1,
        seed: Optional[int] = None,
        canny_low_threshold: Optional[int] = None,
        canny_high_threshold: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Generate fashion design images from a base image and target prompt."""
        if not self.ensure_ready() or self.pipe is None:
            raise RuntimeError(f"Stable Diffusion module is offline: {self.load_error}")

        start_time = time.time()
        self._reset_gpu_peak_memory()
        canny_image = self._preprocess_image(
            base_image_url,
            canny_low_threshold,
            canny_high_threshold,
        )
        prompts = self.build_prompt(target_prompt)
        generated: List[Dict[str, Any]] = []
        requested_images = max(1, min(num_images, settings.MAX_GENERATION_IMAGES))

        for index in range(requested_images):
            image_seed = self._make_image_seed(seed, index)
            generator = self._make_generator(image_seed)
            print(f"[Gen Service] Rendering image {index + 1}/{requested_images} with seed {image_seed}.")
            image = self._render_image(prompts, canny_image, generator)
            generated.append(self._save_generated_image(image, image_seed))
            del image
            self._clear_generation_memory()

        self._record_generation_metrics(start_time, requested_images)
        return generated

    def cleanup_outputs(self) -> int:
        """Delete expired generated output files and return the removal count."""
        ttl_seconds = settings.OUTPUT_TTL_HOURS * 3600
        now = time.time()
        removed = 0
        if ttl_seconds <= 0 or not os.path.isdir(self.output_dir):
            return removed

        for filename in os.listdir(self.output_dir):
            path = os.path.join(self.output_dir, filename)
            if not os.path.isfile(path):
                continue
            if now - os.path.getmtime(path) > ttl_seconds:
                with contextlib.suppress(OSError):
                    os.remove(path)
                    removed += 1
        if removed:
            print(f"[Gen Service] Cleaned {removed} expired output files.")
        return removed


gen_service = GenerativeService()
