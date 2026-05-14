import ipaddress
import socket
from io import BytesIO
from typing import Optional
from urllib.parse import urlparse

import cv2
import numpy as np
import requests
from PIL import Image, ImageOps

from app.core.config import settings


class SafeImageDownloadError(ValueError):
    """Raised when a remote image URL fails safety or image validation."""


def preprocess_canny_edge(
    init_image: Image.Image,
    low_threshold: int = 100,
    high_threshold: int = 200,
) -> Image.Image:
    """Convert a PIL image into a 3-channel Canny edge map for ControlNet.

    Args:
        init_image: Source image received from the user.
        low_threshold: Lower hysteresis threshold passed to ``cv2.Canny``.
        high_threshold: Upper hysteresis threshold passed to ``cv2.Canny``.

    Returns:
        A PIL RGB image where the grayscale NumPy edge mask is stacked into
        three identical channels, matching ControlNet's expected RGB input.
    """
    grayscale_array: np.ndarray = np.array(init_image.convert("L"))
    edge_array: np.ndarray = cv2.Canny(grayscale_array, low_threshold, high_threshold)
    controlnet_array: np.ndarray = np.stack([edge_array, edge_array, edge_array], axis=2)
    return Image.fromarray(controlnet_array)


class ImagePreprocessor:
    """Handles safe image download, resizing, and ControlNet conditioning."""

    def validate_public_url(self, url: str) -> None:
        """Reject non-public or malformed image URLs before downloading."""
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            raise SafeImageDownloadError("Only http/https image URLs are allowed.")
        if not parsed.hostname:
            raise SafeImageDownloadError("Image URL must include a hostname.")

        try:
            addresses = socket.getaddrinfo(parsed.hostname, None)
        except socket.gaierror as exc:
            raise SafeImageDownloadError(f"Cannot resolve image hostname: {parsed.hostname}") from exc

        for _, _, _, _, sockaddr in addresses:
            ip = ipaddress.ip_address(sockaddr[0])
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
        """Download, verify, and decode a public remote image as RGB."""
        self.validate_public_url(url)
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

    def resize_with_padding(
        self,
        image: Image.Image,
        size: Optional[int] = None,
    ) -> Image.Image:
        """Resize an image into a square RGB canvas while preserving aspect ratio."""
        target_size = settings.GENERATION_SIZE if size is None else size
        resized_image = ImageOps.contain(image, (target_size, target_size))
        canvas = Image.new("RGB", (target_size, target_size), (255, 255, 255))
        offset = (
            (target_size - resized_image.width) // 2,
            (target_size - resized_image.height) // 2,
        )
        canvas.paste(resized_image, offset)
        return canvas
