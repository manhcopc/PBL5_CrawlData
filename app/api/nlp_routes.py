from fastapi import APIRouter, HTTPException

from app.schemas.payload import TrendRequest, TrendResponse
from app.services.nlp_service import nlp_service
from app.services.trend_scoring import compute_trend_score, extract_style_keywords

router = APIRouter()


@router.post("/analyze-trend", response_model=TrendResponse)
async def analyze_trend_endpoint(payload: TrendRequest):
    if not nlp_service.ensure_ready():
        raise HTTPException(
            status_code=500,
            detail=f"PhoBERT Service is offline: {nlp_service.load_error}",
        )

    print(f"[API] Analyze trend request {payload.request_id}: {len(payload.products)} products.")
    analyzed_products = []

    for item in payload.products:
        positive_rate = nlp_service.analyze_sentiment(item.reviews) if item.reviews else 0.0
        score = compute_trend_score(item, positive_rate)
        style_keywords = extract_style_keywords(item, payload.category_keyword)

        analyzed_products.append({
            "product_name": item.product_name,
            "source_image_url": str(item.image_url),
            "positive_rate": round(positive_rate, 2),
            "trend_score": score["trend_score"],
            "confidence": score["confidence"],
            "total_reviews": len(item.reviews),
            "style_keywords": style_keywords,
            "scoring_signals": score["signals"],
        })

    analyzed_products.sort(key=lambda product: product["trend_score"], reverse=True)

    return {
        "status": "success",
        "request_id": payload.request_id,
        "trends": analyzed_products[: payload.limit],
    }

