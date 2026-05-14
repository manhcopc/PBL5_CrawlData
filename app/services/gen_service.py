import contextlib
import gc
import ipaddress
import os
import socket
import time
import uuid
from io import BytesIO
from threading import Lock
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests
from PIL import Image, ImageOps

from app.core.config import settings
from app.utils.metrics_logger import log_metric  

class SafeImageDownloadError(ValueError):
    pass

class GenerativeService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GenerativeService, cls).__new__(cls)
            cls._instance._boot()
        return cls._instance

    def _boot(self) -> None:
        self.pipe = None
        self.is_ready = False
        self.load_error: Optional[str] = None
        self.device = "unknown"
        self._load_lock = Lock()
        self.output_dir = settings.OUTPUT_DIR
        os.makedirs(self.output_dir, exist_ok=True)
        self.cleanup_outputs()

    def ensure_ready(self) -> bool:
        if self.is_ready:
            return True

        with self._load_lock:
            if self.is_ready:
                return True

            print("[Gen Service] Lazy-loading Stable Diffusion + ControlNet...")
            try:
                import torch
                from diffusers import (
                    ControlNetModel,
                    StableDiffusionControlNetPipeline,
                    UniPCMultistepScheduler,
                )

                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                print(f"[Gen Service] Device: {self.device}")

                controlnet = ControlNetModel.from_pretrained(
                    "lllyasviel/sd-controlnet-canny",
                    torch_dtype=torch.float16,
                )
                self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5",
                    controlnet=controlnet,
                    torch_dtype=torch.float16,
                )
                self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
                self.pipe.enable_model_cpu_offload()
                self.pipe.enable_xformers_memory_efficient_attention()

                self.is_ready = True
                self.load_error = None
                print("[Gen Service] Stable Diffusion + ControlNet ready.")
                return True
            except Exception as exc:
                self.is_ready = False
                self.load_error = str(exc)
                print(f"[Gen Service] Load failed: {exc}")
                return False

    def _validate_public_url(self, url: str) -> None:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            raise SafeImageDownloadError("Only http/https image URLs are allowed.")
        if not parsed.hostname:
            raise SafeImageDownloadError("Image URL must include a hostname.")

        try:
            addresses = socket.getaddrinfo(parsed.hostname, None)
        except socket.gaierror as exc:
            raise SafeImageDownloadError(f"Cannot resolve image hostname: {parsed.hostname}") from exc

        for family, _, _, _, sockaddr in addresses:
            ip_text = sockaddr[0]
            ip = ipaddress.ip_address(ip_text)
            if (
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                or ip.is_multicast
                or ip.is_reserved
                or ip.is_unspecified
            ):
                raise SafeImageDownloadError("Image URL resolves to a non-public address.")

    def download_image(self, url: str) -> Image.Image:
        self._validate_public_url(url)
        headers = {"User-Agent": "TrendEngine-AI-Core/1.0"}

        with requests.get(
            url,
            headers=headers,
            timeout=settings.IMAGE_DOWNLOAD_TIMEOUT_SECONDS,
            stream=True,
        ) as response:
            response.raise_for_status()
            content_type = response.headers.get("content-type", "").lower()
            if content_type and not content_type.startswith("image/"):
                raise SafeImageDownloadError("URL did not return an image content type.")

            content_length = response.headers.get("content-length")
            if content_length and int(content_length) > settings.MAX_IMAGE_DOWNLOAD_BYTES:
                raise SafeImageDownloadError("Image download is too large.")

            payload = BytesIO()
            downloaded = 0
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue
                downloaded += len(chunk)
                if downloaded > settings.MAX_IMAGE_DOWNLOAD_BYTES:
                    raise SafeImageDownloadError("Image download exceeded size limit.")
                payload.write(chunk)

        payload.seek(0)
        image = Image.open(payload)
        image.verify()
        payload.seek(0)
        image = Image.open(payload).convert("RGB")

        if image.width * image.height > settings.MAX_INPUT_IMAGE_PIXELS:
            raise SafeImageDownloadError("Input image dimensions are too large.")
        return image

    def resize_with_padding(self, image: Image.Image, size: int = settings.GENERATION_SIZE) -> Image.Image:
        image = ImageOps.contain(image, (size, size))
        canvas = Image.new("RGB", (size, size), (255, 255, 255))
        offset = ((size - image.width) // 2, (size - image.height) // 2)
        canvas.paste(image, offset)
        return canvas

    def process_canny_edge(
        self,
        image: Image.Image,
        low_threshold: Optional[int] = None,
        high_threshold: Optional[int] = None,
    ) -> Image.Image:
        import cv2
        import numpy as np

        low = settings.CANNY_LOW_THRESHOLD if low_threshold is None else low_threshold
        high = settings.CANNY_HIGH_THRESHOLD if high_threshold is None else high_threshold
        image_np = np.array(image)
        edges = cv2.Canny(image_np, low, high)
        edges = edges[:, :, None]
        edges = np.concatenate([edges, edges, edges], axis=2)
        return Image.fromarray(edges)

from typing import Dict

    def build_prompt(self, target_prompt: str) -> Dict[str, str]:
        sanitized_prompt = " ".join(target_prompt.split())[:600]
        
        # Bổ sung cụm từ khóa chuyên ép phong cách, chất liệu "Hoàng gia Anh"
        royal_aesthetic = (
            "British royal fashion aesthetic, Savile Row tailoring, "
            "premium velvet and heavy tweed materials, intricate gold button details, "
            "regal aristocratic vibe, elegant luxury menswear"
        )
        
        # Base prompt giữ nguyên để đảm bảo chất lượng ảnh chụp
        base_prompt = (
            "high-end fashion product photography, sharp tailoring, realistic fabric texture, "
            "studio lighting, detailed garment construction, clean background, 8k resolution"
        )
        
        # Thêm "cheap fabric" (vải rẻ tiền) vào negative prompt để cấm AI vẽ hàng chợ
        negative_prompt = (
            "low quality, blurry, distorted garment, messy seams, bad anatomy, extra limbs, "
            "text, watermark, logo artifacts, cropped garment, duplicate clothing, cheap fabric"
        )
        
        return {
            # Nối chuỗi: [Yêu cầu người dùng] + [Phong cách Hoàng gia] + [Chất lượng ảnh]
            "prompt": f"{sanitized_prompt}, {royal_aesthetic}, {base_prompt}",
            "negative_prompt": negative_prompt,
        }

    def _public_output_url(self, filename: str) -> str:
        return f"{settings.PUBLIC_BASE_URL.rstrip('/')}/outputs/{filename}"

    def _make_generator(self, seed: int):
        import torch

        if self.device == "cuda":
            return torch.Generator(device="cuda").manual_seed(seed)
        return torch.Generator(device="cpu").manual_seed(seed)

    def generate_design(
        self,
        base_image_url: str,
        target_prompt: str,
        num_images: int = 1,
        seed: Optional[int] = None,
        canny_low_threshold: Optional[int] = None,
        canny_high_threshold: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if not self.ensure_ready() or self.pipe is None:
            raise RuntimeError(f"Stable Diffusion module is offline: {self.load_error}")

        import torch

        # --- BẮT ĐẦU ĐO LƯỜNG ---
        start_time = time.time()
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        print(f"[Gen Service] Downloading base image: {base_image_url}")
        base_image = self.resize_with_padding(self.download_image(base_image_url))
        canny_image = self.process_canny_edge(
            base_image,
            low_threshold=canny_low_threshold,
            high_threshold=canny_high_threshold,
        )
        prompts = self.build_prompt(target_prompt)

        generated: List[Dict[str, Any]] = []
        requested_images = max(1, min(num_images, settings.MAX_GENERATION_IMAGES))

        for index in range(requested_images):
            image_seed = seed + index if seed is not None else int.from_bytes(os.urandom(4), "big")
            generator = self._make_generator(image_seed)
            print(f"[Gen Service] Rendering image {index + 1}/{requested_images} with seed {image_seed}.")

            autocast_context = (
                torch.autocast("cuda") if self.device == "cuda" else contextlib.nullcontext()
            )
            with torch.inference_mode(), autocast_context:
                image = self.pipe(
                    prompt=prompts["prompt"],
                    negative_prompt=prompts["negative_prompt"],
                    image=canny_image,
                    controlnet_conditioning_scale=settings.CONTROLNET_CONDITIONING_SCALE,
                    num_inference_steps=settings.GENERATION_STEPS,
                    guidance_scale=settings.GENERATION_GUIDANCE_SCALE,
                    generator=generator,
                ).images[0]

            filename = f"design_{uuid.uuid4().hex[:8]}.png"
            local_path = os.path.join(self.output_dir, filename)
            image.save(local_path)
            generated.append({
                "url": self._public_output_url(filename),
                "seed": image_seed,
            })
            print(f"[Gen Service] Saved: {local_path}")
            del image
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # --- KẾT THÚC ĐO LƯỜNG VÀ GHI LOG ---
        end_time = time.time()
        latency = end_time - start_time
        peak_vram = 0.0

        if self.device == "cuda":
            peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
        
        log_metric(
            api_name="generate_design", 
            latency_sec=latency, 
            vram_gb=peak_vram, 
            extra_info=f"num_images:{requested_images}"
        )
        print(f"[Metrics] GenAI xong trong {latency:.2f}s | Đỉnh VRAM: {peak_vram:.2f} GB")

        return generated

    def cleanup_outputs(self) -> int:
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