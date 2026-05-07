from playwright.sync_api import sync_playwright, TimeoutError
import random
import time
import re
import json
from pathlib import Path
from urllib.parse import urljoin, urlparse, parse_qs, urlsplit, urlunsplit
from typing import List, Dict, Any
from datetime import datetime, timezone, timedelta

# ==============================================
# CONFIG
# ==============================================

LINK = "https://www.lazada.vn/tag/%C4%91%E1%BB%93-vest/?spm=a2o4n.homepage.search.d_go&q=%C4%91%E1%BB%93%20vest&catalog_redirect_tag=true"
MAX_PRODUCTS = 100  # None = no limit
SAVE_REVIEWS = False  # Disable review crawl; use trend_analyzer.py instead
MAX_REVIEWS_PER_PRODUCT = 0

LAZADA_BASE = "https://www.lazada.vn"
VN_TZ = timezone(timedelta(hours=7))

# ==============================================
# HELPER: Extract search name from URL
# ==============================================

def extract_search_name_from_url(url: str) -> str:
    """
    Extract search name from URL parameter 'q=' and convert to safe folder name.
    Examples:
    - https://...?q=quần%20nam -> "quần_nam"
    - https://...?q=áo+nữ -> "áo_nữ"
    
    Returns:
        Sanitized search name (safe for folder names)
    """
    try:
        parsed = urlparse(url)
        params = parse_qs(parsed.query)
        if 'q' in params and params['q']:
            search_term = params['q'][0]
            # URL decode
            search_term = search_term.replace('%20', ' ').replace('+', ' ')
            search_term = search_term.strip()
            # Sanitize for folder name
            search_term = re.sub(r"[^\w\s\.-]", "_", search_term, flags=re.UNICODE)
            search_term = re.sub(r"\s+", "_", search_term).strip("_")
            return search_term[:120] if search_term else "unknown_search"
    except Exception:
        pass
    return "unknown_search"


# ==============================================
# OUTPUT PATHS (organized by search term)
# ==============================================

SCRIPT_DIR = Path(__file__).parent.resolve()  # /Users/copc/Sync/SCHOOL/HocKi6/PBL5/dataset/lazada

# Extract search name and create subfolder
SEARCH_NAME = extract_search_name_from_url(LINK)
DATASET_DIR = SCRIPT_DIR / SEARCH_NAME  # lazada/<search_name>/
DATASET_DIR.mkdir(parents=True, exist_ok=True)

IMAGES_DIR = DATASET_DIR / "lazada_images"  # lazada/<search_name>/lazada_images/
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

PRODUCTS_JSON = DATASET_DIR / "lazada_products.json"
PRODUCT_SNAPSHOTS_JSONL = DATASET_DIR / "lazada_product_snapshots.jsonl"
REVIEWS_JSONL = DATASET_DIR / "lazada_reviews.jsonl"

print(f"[Setup] Search name: '{SEARCH_NAME}'")
print(f"[Setup] Data directory: {DATASET_DIR}")

# ==============================================
# UTILITY FUNCTIONS
# ==============================================

def safe_filename(s: str, max_len: int = 120) -> str:
    s = re.sub(r"[^\w\s\.-]", "_", s, flags=re.UNICODE)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s[:max_len] if len(s) > max_len else s


def build_page_url(base_url, page_num):
    """Build pagination URL by modifying page parameter."""
    if page_num == 1:
        return base_url

    if "page=" in base_url:
        return re.sub(r"page=\d+", f"page={page_num}", base_url)

    separator = "&" if "?" in base_url else "?"
    return f"{base_url}{separator}page={page_num}"


def get_text(page, selectors) -> str:
    for selector in selectors:
        try:
            element = page.locator(selector).first
            element.wait_for(timeout=1500)
            text = element.inner_text().strip()
            if text:
                return text
        except Exception:
            continue
    return "N/A"


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def append_review_jsonl(review: Dict[str, Any]) -> None:
    append_jsonl(REVIEWS_JSONL, review)


def append_product_snapshot_jsonl(snapshot: Dict[str, Any]) -> None:
    append_jsonl(PRODUCT_SNAPSHOTS_JSONL, snapshot)


def _infer_product_id_from_url(u: str) -> str:
    m = re.search(r"-i(\d+)\.html", u)
    return m.group(1) if m else safe_filename(urlparse(u).path or "unknown")


def extract_image_urls(page) -> list[str]:
    urls = set()
    img_selectors = ["div.seo-gallery-hidden img"]

    for sel in img_selectors:
        try:
            imgs = page.query_selector_all(sel)
        except Exception:
            imgs = []

        for img in imgs:
            for attr in ("src", "data-src", "data-ks-lazyload", "data-original"):
                try:
                    v = img.get_attribute(attr)
                except Exception:
                    v = None
                if v and v.startswith(("http://", "https://", "//")):
                    urls.add("https:" + v if v.startswith("//") else v)

            try:
                srcset = img.get_attribute("srcset")
            except Exception:
                srcset = None
            if srcset:
                parts = [p.strip().split(" ")[0] for p in srcset.split(",") if p.strip()]
                for v in parts:
                    if v.startswith(("http://", "https://", "//")):
                        urls.add("https:" + v if v.startswith("//") else v)

    return sorted({u for u in urls if u and "data:image" not in u})


def download_image_via_playwright(request_ctx, url: str, out_path: Path) -> bool:
    try:
        resp = request_ctx.get(
            url,
            headers={
                "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
                "Referer": "https://www.lazada.vn/",
            },
            timeout=30000,
        )
        if not resp.ok:
            return False

        ctype = (resp.headers.get("content-type") or "").lower()
        if "image" not in ctype:
            return False

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(resp.body())
        return True
    except Exception:
        return False


# ==============================================
# SOLD + RATING PARSING
# ==============================================

def parse_sold_count_from_text(txt: str) -> int:
    if not isinstance(txt, str) or not txt.strip():
        return 0
    t = txt.lower().replace(",", ".")

    m = re.search(r"(?:đã\s*bán|da\s*ban|sold)\s*([0-9]+(?:\.[0-9]+)?)(\s*[km])?", t)
    if not m:
        m = re.search(r"([0-9]+(?:\.[0-9]+)?)(\s*[km])?\s*sold", t)
    if not m:
        return 0

    num = float(m.group(1))
    suf = (m.group(2) or "").strip()
    if suf == "k":
        num *= 1000
    elif suf == "m":
        num *= 1_000_000
    return int(num)


def get_sold_count_from_pdp(page) -> int:
    selectors = [
        "a.pdp-review-link",
        "div.pdp-review-summary",
        "div.pdp-mod-product-info",
        "div:has-text('Đã bán')",
        "span:has-text('Đã bán')",
        "div:has-text('sold')",
        "span:has-text('sold')",
    ]

    for sel in selectors:
        try:
            loc = page.locator(sel).first
            if loc.count():
                txt = (loc.inner_text() or "").strip()
                v = parse_sold_count_from_text(txt)
                if v > 0:
                    return v
        except Exception:
            pass

    try:
        html = page.content()
        v = parse_sold_count_from_text(html)
        if v > 0:
            return v
    except Exception:
        pass

    return 0


def parse_rating_from_text(txt: str) -> tuple[float, int]:
    """Return (rating_score, rating_total) from PDP rating text."""
    if not isinstance(txt, str) or not txt.strip() or txt == "N/A":
        return 0.0, 0

    t = " ".join(txt.replace("\n", " ").split())

    score = 0.0
    total = 0

    m_score = re.search(r"(\d+(?:[\.,]\d+)?)\s*(?:/\s*5)?", t)
    if m_score:
        try:
            score = float(m_score.group(1).replace(",", "."))
        except Exception:
            score = 0.0

    m_total = re.search(r"\(\s*([0-9][0-9\.,\s]*)\s*\)", t)
    if not m_total:
        m_total = re.search(r"([0-9][0-9\.,\s]*)\s*(?:ratings?|đánh\s*giá|nhận\s*xét|reviews?)\b", t, flags=re.IGNORECASE)

    if m_total:
        raw = m_total.group(1).replace(" ", "")
        raw = raw.replace(",", "").replace(".", "")
        try:
            total = int(raw)
        except Exception:
            total = 0

    if not (0.0 <= score <= 5.0):
        score = 0.0

    return score, total


def get_pdp_rating(page) -> tuple[float, int]:
    """Extract PDP rating score and total count.

    New Lazada UI often renders:
      <span class="container-star-v2-score">4.8</span>
      <span class="container-star-v2-count">(253)</span>

    This avoids parsing SVG-heavy containers.
    """
    # Prefer explicit score/count elements
    try:
        score_raw = page.locator("span.container-star-v2-score").first.inner_text().strip()
        if score_raw:
            try:
                score = float(score_raw.replace(",", "."))
            except Exception:
                score = 0.0

            count_raw = ""
            try:
                count_raw = page.locator("span.container-star-v2-count").first.inner_text().strip()
            except Exception:
                count_raw = ""

            total = 0
            if count_raw:
                m = re.search(r"([0-9][0-9\.,]*)", count_raw)
                if m:
                    raw = m.group(1).replace(",", "").replace(".", "")
                    try:
                        total = int(raw)
                    except Exception:
                        total = 0

            if 0.0 <= score <= 5.0:
                return score, total
    except Exception:
        pass

    # Fallback: previous text-based selectors
    rating_text = get_text(page, ["a.pdp-review-link", "div.summary-rating-title", "div.pdp-review-summary"])
    return parse_rating_from_text(rating_text)


# ==============================================
# PAGINATION
# ==============================================

def crawl_all_products_with_pagination(page, request_ctx, base_url: str, max_products: int = None) -> tuple[list[str], dict]:
    """
    Crawl product URLs from listing page with automatic pagination.
    """
    print("\n[Pagination] Starting automatic pagination...")
    
    product_urls = []
    product_sold_map = {}
    page_num = 0
    max_pages = 20  # Safety limit
    
    while page_num < max_pages:
        page_num += 1
        
        if max_products and len(product_urls) >= max_products:
            print(f"[Pagination] Reached limit ({max_products}). Stopping.")
            break
        
        url = build_page_url(base_url, page_num)
        print(f"\n[Pagination] Page {page_num}: {url}")
        
        try:
            page.goto(url, timeout=60000)
            page.wait_for_load_state("networkidle")
            
            # Scroll để load sản phẩm
            for _ in range(5):
                page.mouse.wheel(0, 2000)
                page.wait_for_timeout(800)
            
            # Lấy product cards
            product_elements = page.query_selector_all("div[data-qa-locator='product-item']")
            print(f"  Found {len(product_elements)} products on this page")
            
            if not product_elements:
                print(f"[Pagination] No products found. Stopping.")
                break
            
            new_urls_this_page = 0
            for item in product_elements:
                try:
                    link_element = item.query_selector("a")
                    url_item = link_element.get_attribute("href") if link_element else None
                    
                    if url_item and url_item not in product_urls:
                        product_urls.append(url_item)
                        new_urls_this_page += 1
                        
                        # Parse sold từ listing card
                        try:
                            txt = (item.inner_text() or "").strip()
                            sold = parse_sold_count_from_text(txt)
                            if sold > 0:
                                product_sold_map[url_item] = sold
                        except Exception:
                            pass
                        
                        if max_products and len(product_urls) >= max_products:
                            break
                except Exception:
                    continue
            
            print(f"  Added {new_urls_this_page} new. Total: {len(product_urls)}")
            
            if new_urls_this_page == 0:
                print(f"[Pagination] No new products found. Stopping.")
                break
        
        except TimeoutError:
            print(f"  Timeout on page {page_num}. Stopping.")
            break
        except Exception as e:
            print(f"  Error on page {page_num}: {e}. Stopping.")
            break
    
    product_urls = list(dict.fromkeys(product_urls))
    print(f"\n[Pagination] Completed! Total unique: {len(product_urls)}")
    
    return product_urls, product_sold_map


# ==============================================
# MAIN
# ==============================================

with sync_playwright() as p:
    print("[1/6] Connecting to Chrome via CDP at http://localhost:9022 ...")
    try:
        browser = p.chromium.connect_over_cdp("http://localhost:9022")
    except Exception as e:
        raise SystemExit(
            "Cannot connect to CDP on http://localhost:9022. "
            "Start Chrome with: --remote-debugging-port=9022 --user-data-dir=/path/to/profile\n"
            f"Original error: {e}"
        )

    context = browser.contexts[0] if browser.contexts else browser.new_context()
    page = context.pages[0] if context.pages else context.new_page()
    request_ctx = context.request

    print("[2/6] Opening search/tag page ...")
    page.goto(LINK, timeout=60000)
    page.wait_for_timeout(5000)

    print("[3/6] Crawling products with automatic pagination ...")
    product_urls, product_sold_map = crawl_all_products_with_pagination(
        page, request_ctx, LINK, max_products=MAX_PRODUCTS
    )

    if MAX_PRODUCTS is not None:
        product_urls = product_urls[:MAX_PRODUCTS]

    print(f"[4/6] Processing {len(product_urls)} products...")

    products_out: list[dict] = []

    for idx, product_url in enumerate(product_urls):
        try:
            if product_url.startswith("/"):
                absolute_url = urljoin(page.url, product_url)
            elif product_url.startswith("//"):
                absolute_url = "https:" + product_url
            else:
                absolute_url = product_url

            print(f"\n[{idx + 1}/{len(product_urls)}] Processing: {absolute_url}")

            sold_count = product_sold_map.get(product_url) or product_sold_map.get(absolute_url) or 0

            page.goto(absolute_url, wait_until="domcontentloaded", timeout=60000)

            try:
                page.wait_for_selector("#module_product_price_1, #module_product_detail", timeout=20000)
            except TimeoutError:
                print("  - Skip: product page did not load required modules in time")
                continue

            product_name = get_text(page, [".pdp-product-title h1", "h1.pdp-mod-product-badge-title"])
            shop_name = get_text(page, ["a[href*='/shop/']", "div[class*='seller-name-v2__top'] a"])

            color_elements = page.query_selector_all("span.sku-variable-img-name")
            colors = list(set([c.get_attribute("title") for c in color_elements if c.get_attribute("title")]))
            color_str = ", ".join(colors) if colors else "N/A"

            price_text = get_text(
                page,
                [
                    "span.pdp-v2-product-price-content-salePrice-amount",
                    "div[class*='pdp-price']",
                    "span[class*='Price--priceText']",
                    ".pdp-price_type_normal",
                ],
            )

            rating_score, rating_total = get_pdp_rating(page)

            if sold_count == 0:
                sold_count = get_sold_count_from_pdp(page)

            image_urls = extract_image_urls(page)
            product_id = _infer_product_id_from_url(absolute_url)

            folder = IMAGES_DIR / str(product_id)
            folder.mkdir(parents=True, exist_ok=True)

            base = safe_filename(product_name if product_name != "N/A" else product_id)

            image_paths: list[str] = []
            for i_img, img_url in enumerate(image_urls, start=1):
                out_path = folder / f"{base}_{i_img}.jpg"
                if out_path.exists():
                    image_paths.append(str(out_path))
                    continue
                if download_image_via_playwright(request_ctx, img_url, out_path):
                    image_paths.append(str(out_path))

            reviews_written = 0

            try:
                price_number = float(re.sub(r"[^0-9]", "", price_text)) if price_text not in ("N/A", "") else 0.0
            except Exception:
                price_number = 0.0

            crawled_at = datetime.now(VN_TZ).isoformat()

            product_obj = {
                "product_id": str(product_id),
                "product_name": product_name,
                "color": color_str,
                "price": price_number,
                "shop_name": shop_name,
                "rating": float(rating_score) if rating_score else 0.0,
                "rating_score": float(rating_score) if rating_score else 0.0,
                "rating_total": int(rating_total) if rating_total else 0,
                "sold": int(sold_count),
                "product_url": absolute_url,
                "image_urls": image_urls,
                "image_paths": image_paths,
                "crawled_at": crawled_at,
            }
            products_out.append(product_obj)

            append_product_snapshot_jsonl(
                {
                    "product_id": str(product_id),
                    "product_url": absolute_url,
                    "product_name": product_name,
                    "shop_name": shop_name,
                    "price": price_number,
                    "rating": float(rating_score) if rating_score else 0.0,
                    "rating_score": float(rating_score) if rating_score else 0.0,
                    "rating_total": int(rating_total) if rating_total else 0,
                    "sold": int(sold_count),
                    "reviews_written": int(reviews_written),
                    "crawled_at": crawled_at,
                }
            )

            print(f"✔ {product_name} | Sold: {sold_count} | Rating: {rating_score}/{rating_total}")

        except Exception as e:
            print(f"  - Error: {e}")
            continue

    print(f"\n[6/6] Writing products JSON ({len(products_out)} items) ...")
    if products_out:
        write_json(PRODUCTS_JSON, products_out)
        print(f"Done. Wrote: {PRODUCTS_JSON}")
        print(f"Time-series snapshots: {PRODUCT_SNAPSHOTS_JSONL}")
        print(f"\nNext step: Run trend_analyzer.py to identify trending products.")
    else:
        print("Done. No products collected.")
