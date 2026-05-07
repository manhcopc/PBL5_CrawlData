from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, HttpUrl, conint, confloat, constr, validator

from app.core.config import settings


NonEmptyText = constr(strip_whitespace=True, min_length=1)


class ScrapedProduct(BaseModel):
    product_name: NonEmptyText = Field(..., max_length=240)
    image_url: HttpUrl
    reviews: List[NonEmptyText] = Field(default_factory=list, max_items=200)
    scenario: Optional[constr(strip_whitespace=True, max_length=40)] = None
    sales_velocity: Optional[confloat(ge=0)] = None
    created_at: Optional[datetime] = None


class TrendRequest(BaseModel):
    request_id: NonEmptyText = Field(..., max_length=80)
    category_keyword: NonEmptyText = Field(..., max_length=80)
    products: List[ScrapedProduct] = Field(..., min_items=1, max_items=100)
    limit: conint(ge=1, le=20) = 5


class TrendProduct(BaseModel):
    product_name: str
    source_image_url: str
    positive_rate: float
    trend_score: float
    confidence: float
    total_reviews: int
    style_keywords: List[str]
    scoring_signals: Dict[str, Any]


class TrendResponse(BaseModel):
    status: str
    request_id: str
    trends: List[TrendProduct]


class DesignRequest(BaseModel):
    request_id: NonEmptyText = Field(..., max_length=80)
    target_style_prompt: NonEmptyText = Field(..., max_length=600)
    base_image_url: HttpUrl
    num_images: conint(ge=1, le=settings.MAX_GENERATION_IMAGES) = 1
    seed: Optional[conint(ge=0, le=2**32 - 1)] = None
    canny_low_threshold: Optional[conint(ge=0, le=255)] = None
    canny_high_threshold: Optional[conint(ge=0, le=255)] = None

    @validator("target_style_prompt")
    def normalize_prompt(cls, value: str) -> str:
        return " ".join(value.split())

    @validator("canny_high_threshold")
    def validate_canny_order(cls, high: Optional[int], values: Dict[str, Any]) -> Optional[int]:
        low = values.get("canny_low_threshold")
        if high is not None and low is not None and high <= low:
            raise ValueError("canny_high_threshold must be greater than canny_low_threshold")
        return high


class GeneratedDesign(BaseModel):
    url: str
    seed: int
    filename: str


class GenerationJobAccepted(BaseModel):
    status: str
    request_id: str
    job_id: str


class GenerationJobStatus(BaseModel):
    job_id: str
    request_id: str
    status: str
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    generated_designs: List[GeneratedDesign] = Field(default_factory=list)
    error: Optional[str] = None

