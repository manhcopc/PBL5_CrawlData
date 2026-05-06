import os

class Settings:
    PROJECT_NAME: str = "TrendEngine AI Core"
    VERSION: str = "1.0.0"
    
    # Lấy đường dẫn gốc của project
    ROOT_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Cấu hình đường dẫn Model
    PHOBERT_PATH: str = os.path.join(ROOT_DIR, "data", "vest", "output", "models", "phobert")
    
    # Cấu hình ngưỡng Khen/Chê
    TREND_THRESHOLD: float = 0.7

    # Colab/ngrok runtime controls
    PUBLIC_BASE_URL: str = os.getenv("PUBLIC_BASE_URL", "http://localhost:8000")
    OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", os.path.join(ROOT_DIR, "outputs"))
    OUTPUT_TTL_HOURS: int = int(os.getenv("OUTPUT_TTL_HOURS", "24"))

    # T4-safe generation defaults
    MAX_GENERATION_IMAGES: int = int(os.getenv("MAX_GENERATION_IMAGES", "2"))
    GENERATION_SIZE: int = int(os.getenv("GENERATION_SIZE", "512"))
    GENERATION_STEPS: int = int(os.getenv("GENERATION_STEPS", "30"))
    GENERATION_GUIDANCE_SCALE: float = float(os.getenv("GENERATION_GUIDANCE_SCALE", "7.5"))
    CONTROLNET_CONDITIONING_SCALE: float = float(os.getenv("CONTROLNET_CONDITIONING_SCALE", "0.8"))
    CANNY_LOW_THRESHOLD: int = int(os.getenv("CANNY_LOW_THRESHOLD", "100"))
    CANNY_HIGH_THRESHOLD: int = int(os.getenv("CANNY_HIGH_THRESHOLD", "200"))

    # Safe remote image download limits
    IMAGE_DOWNLOAD_TIMEOUT_SECONDS: int = int(os.getenv("IMAGE_DOWNLOAD_TIMEOUT_SECONDS", "15"))
    MAX_IMAGE_DOWNLOAD_BYTES: int = int(os.getenv("MAX_IMAGE_DOWNLOAD_BYTES", str(8 * 1024 * 1024)))
    MAX_INPUT_IMAGE_PIXELS: int = int(os.getenv("MAX_INPUT_IMAGE_PIXELS", str(4096 * 4096)))

    # One worker keeps VRAM stable on a single Tesla T4.
    GENERATION_QUEUE_CONCURRENCY: int = int(os.getenv("GENERATION_QUEUE_CONCURRENCY", "1"))
    GENERATION_JOB_TTL_HOURS: int = int(os.getenv("GENERATION_JOB_TTL_HOURS", "6"))

settings = Settings()
