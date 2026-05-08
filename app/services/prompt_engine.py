import re
from typing import Iterable, List


HEAVY_MATERIAL_TERMS = (
    "leather",
    "fur",
    "heavy wool",
    "wool",
    "velvet",
    "tweed",
    "fleece",
    "thick knit",
)

SUMMER_TERMS = (
    "breathable linen",
    "lightweight fabric",
    "airy silhouette",
    "bright pastel colors",
    "sun-ready styling",
)

WINTER_TERMS = (
    "warm",
    "layered",
    "heavy fabric",
    "darker tones",
    "insulated texture",
)

GEN_Z_TERMS = (
    "streetwear aesthetic",
    "trendy vibe",
    "bold cuts",
    "expressive styling",
)

OFFICE_WORKER_TERMS = (
    "professional",
    "elegant",
    "business casual",
    "polished tailoring",
)

QUALITY_TERMS = (
    "commercially viable fashion design",
    "realistic fabric detail",
    "cohesive garment styling",
)


def extract_prompt_keywords(prompt: str) -> List[str]:
    """Split an existing prompt into reusable style phrases."""
    candidates = re.split(r"[,;|]+", prompt)
    return [_clean_phrase(candidate) for candidate in candidates if _clean_phrase(candidate)]


def build_dynamic_prompt(base_style: str, keywords: list[str], season: str, audience: str) -> str:
    normalized_season = _normalize_context(season)
    normalized_audience = _normalize_context(audience)
    blocked_terms: Iterable[str] = ()
    context_terms: List[str] = []
    negative_guidance: List[str] = []

    if _matches_any(normalized_season, ("summer", "hot")):
        blocked_terms = HEAVY_MATERIAL_TERMS
        context_terms.extend(SUMMER_TERMS)
        negative_guidance.append(
            "avoid heavy materials such as leather, fur, heavy wool, velvet, tweed, and fleece"
        )
    elif "winter" in normalized_season:
        context_terms.extend(WINTER_TERMS)

    if "gen z" in normalized_audience or "gen-z" in normalized_audience or "genz" in normalized_audience:
        context_terms.extend(GEN_Z_TERMS)
    elif "office worker" in normalized_audience or "business" in normalized_audience:
        context_terms.extend(OFFICE_WORKER_TERMS)

    base_term = _remove_blocked_terms(_clean_phrase(base_style), blocked_terms)
    filtered_style_terms = [base_term] if base_term else []
    filtered_style_terms.extend(_filter_keyword_terms(keywords, blocked_terms))
    prompt_parts = _dedupe_phrases(
        [
            *filtered_style_terms,
            *context_terms,
            *QUALITY_TERMS,
            *negative_guidance,
        ]
    )

    return ", ".join(prompt_parts)


def _filter_keyword_terms(terms: Iterable[str], blocked_terms: Iterable[str]) -> List[str]:
    filtered = []
    for term in terms:
        cleaned = _clean_phrase(term)
        if _contains_blocked_term(cleaned, blocked_terms):
            continue
        if cleaned:
            filtered.append(cleaned)
    return filtered


def _remove_blocked_terms(value: str, blocked_terms: Iterable[str]) -> str:
    cleaned = value
    for blocked in blocked_terms:
        pattern = r"\b" + re.escape(blocked) + r"\b"
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    return _clean_phrase(cleaned)


def _contains_blocked_term(value: str, blocked_terms: Iterable[str]) -> bool:
    return any(
        re.search(r"\b" + re.escape(blocked) + r"\b", value, flags=re.IGNORECASE)
        for blocked in blocked_terms
    )


def _clean_phrase(value: str) -> str:
    cleaned = " ".join(str(value).split())
    cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned)
    cleaned = re.sub(r"^[\s,.;:/-]+|[\s,.;:/-]+$", "", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned


def _dedupe_phrases(phrases: Iterable[str]) -> List[str]:
    seen = set()
    deduped = []
    for phrase in phrases:
        cleaned = _clean_phrase(phrase)
        key = cleaned.casefold()
        if cleaned and key not in seen:
            seen.add(key)
            deduped.append(cleaned)
    return deduped


def _normalize_context(value: str) -> str:
    return _clean_phrase(value).casefold()


def _matches_any(value: str, options: Iterable[str]) -> bool:
    return any(option in value for option in options)
