import math
import re
import unicodedata
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from app.core.config import settings
from app.schemas.payload import ScrapedProduct


FASHION_KEYWORDS = {
    "colors": [
        "black", "white", "red", "blue", "green", "brown", "beige", "pastel",
        "den", "trang", "do", "xanh", "nau", "kem", "be", "pastel",
    ],
    "materials": [
        "velvet", "denim", "leather", "cotton", "linen", "silk", "wool",
        "nhung", "jean", "da", "cotton", "lanh", "lua", "len", "kaki",
    ],
    "fits": [
        "oversized", "slim fit", "regular fit", "cropped", "wide leg",
        "form rong", "hack dang", "om body", "dang suong", "dai tay",
    ],
    "styles": [
        "vintage", "streetwear", "y2k", "minimal", "classic", "tailcoat",
        "vintage", "duong pho", "toi gian", "co dien", "thanh lich",
        "vest", "blazer", "suit", "ao khoac", "set do",
    ],
}


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFD", value.lower())
    without_marks = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    return re.sub(r"\s+", " ", without_marks).strip()


def extract_style_keywords(product: ScrapedProduct, category_keyword: str, limit: int = 8) -> List[str]:
    text_parts = [product.product_name, category_keyword, *product.reviews[:20]]
    haystack = _normalize_text(" ".join(text_parts))
    found: List[str] = []

    for terms in FASHION_KEYWORDS.values():
        for term in terms:
            normalized = _normalize_text(term)
            if normalized in haystack and normalized not in found:
                found.append(normalized)
            if len(found) >= limit:
                return found

    fallback = _normalize_text(category_keyword)
    return found or ([fallback] if fallback else [])


def _sales_signal(scenario: Optional[str], sales_velocity: Optional[float]) -> float:
    scenario_score = 1.0 if scenario and scenario.lower() == "trending" else 0.0
    velocity_score = 0.0
    if sales_velocity is not None:
        velocity_score = min(1.0, math.log1p(sales_velocity) / math.log1p(100.0))
    return max(scenario_score, velocity_score)


def _freshness_signal(created_at: Optional[datetime]) -> float:
    if created_at is None:
        return 0.5

    now = datetime.now(timezone.utc)
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)

    age_days = max(0.0, (now - created_at).total_seconds() / 86400.0)
    return max(0.0, min(1.0, 1.0 - (age_days / 30.0)))


def compute_trend_score(product: ScrapedProduct, positive_rate: float) -> Dict[str, Any]:
    total_reviews = len(product.reviews)
    confidence = min(1.0, math.log1p(total_reviews) / math.log1p(20.0))
    sales = _sales_signal(product.scenario, product.sales_velocity)
    freshness = _freshness_signal(product.created_at)

    trend_score = (
        positive_rate * 0.65
        + confidence * 0.20
        + sales * 0.10
        + freshness * 0.05
    )

    return {
        "trend_score": round(trend_score, 4),
        "confidence": round(confidence, 4),
        "signals": {
            "sentiment": round(positive_rate, 4),
            "review_confidence": round(confidence, 4),
            "sales": round(sales, 4),
            "freshness": round(freshness, 4),
            "threshold": settings.TREND_THRESHOLD,
        },
    }

