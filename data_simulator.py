"""
Data Simulation Module - Theo Solution.txt
Lọc 100 sản phẩm tiêu biểu (50 hot + 50 ế), gắn nhãn scenario, sinh sales_history.
Tất cả paths là relative (tương đối) để dễ chia sẻ.
"""

import json
import csv
import random
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np

# Relative paths - tương đối so với thư mục script này (lazada folder)
SCRIPT_DIR = Path(__file__).parent.resolve()
PRODUCTS_JSON = SCRIPT_DIR / "đồ_vest" / "lazada_products.json"  # Input từ crawlData.py cùng folder
OUTPUT_DIR = SCRIPT_DIR / "đồ_vest" / "output"  # Tạo folder output bên trong lazada/
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SIM_DIR = OUTPUT_DIR / "simulation"
SIM_DIR.mkdir(parents=True, exist_ok=True)

FILTERED_PRODUCTS_JSON = SIM_DIR / "filtered_products_100.json"
SALES_HISTORY_CSV = SIM_DIR / "sales_history.csv"

# Keywords để gắn nhãn "Trending" (~5 sản phẩm)
TRENDING_KEYWORDS = ["baby tee", "áo baby tee", "crop top", "croptop", "trendy"]


def load_and_validate_products() -> list[dict]:
    """
    Load từ lazada_products.json và kiểm tra fields cần thiết.
    Fields cần có theo Solution.txt: title, soldCnt, price, image, ratingNumber
    """
    if not PRODUCTS_JSON.exists():
        print(f"❌ Error: {PRODUCTS_JSON} không tồn tại. Chạy crawlData.py trước!")
        print(f"   Expected path: {PRODUCTS_JSON.absolute()}")
        return []

    with open(PRODUCTS_JSON, "r", encoding="utf-8") as f:
        products = json.load(f)

    if not products:
        print(f"❌ Error: {PRODUCTS_JSON} rỗng!")
        return []

    # Kiểm tra và chuẩn hoá fields theo Solution.txt
    validated = []
    for p in products:
        if not p.get("product_id") or not p.get("product_name"):
            continue

        normalized = {
            "product_id": str(p.get("product_id")),
            "title": p.get("product_name", ""),  # Tên sản phẩm
            "soldCnt": int(p.get("sold", 0)),  # Tổng số lượng đã bán
            "price": float(p.get("price", 0)),  # Giá bán
            "image": (p.get("image_urls") or [""])[0] if p.get("image_urls") else "",  # Link ảnh
            "ratingNumber": int(p.get("rating_total", 0)),  # Tổng số lượt đánh giá
            "rating_score": float(p.get("rating_score", 0)),
            "shop_name": p.get("shop_name", ""),
            "product_url": p.get("product_url", ""),
        }
        validated.append(normalized)

    return validated


def filter_and_tag_products(all_products: list[dict]) -> tuple[list[dict], list[str]]:
    """
    Bước 1-2 (theo Solution.txt):
    
    Bước 1: Lọc dữ liệu
    - Lọc ra khoảng 100 sản phẩm tiêu biểu
    - 50 sản phẩm bán chạy (top by soldCnt)
    - 50 sản phẩm bán ế (bottom by soldCnt)
    
    Bước 2: Gắn nhãn kịch bản (Scenario Tagging)
    - Gắn nhãn "Trending" cho khoảng 5 sản phẩm
    - Gắn nhãn "Normal" cho các sản phẩm còn lại

    Returns:
        (filtered_products_100, trending_product_ids)
    """
    if len(all_products) < 100:
        print(f"⚠️  Chỉ có {len(all_products)} sản phẩm, cần ít nhất 100. Dùng hết.")
        hot_50 = all_products[: len(all_products) // 2]
        normal_50 = all_products[len(all_products) // 2 :]
    else:
        # Sort by soldCnt descending
        sorted_prods = sorted(all_products, key=lambda x: x.get("soldCnt", 0), reverse=True)
        hot_50 = sorted_prods[:50]  # 50 cái bán chạy
        normal_50 = sorted_prods[-50:]  # 50 cái bán ế

    filtered = hot_50 + normal_50

    # Bước 2: Gắn nhãn "Trending" cho ~5 sản phẩm
    # Ưu tiên từ hot_50, match keywords
    trending_ids = []
    for prod in hot_50[:15]:  # Check top 15 hot products
        name_lower = (prod.get("title") or "").lower()
        if any(kw in name_lower for kw in TRENDING_KEYWORDS):
            trending_ids.append(prod.get("product_id"))
            if len(trending_ids) >= 5:
                break

    # Nếu không đủ 5, pick random từ hot_50
    while len(trending_ids) < 5 and len(hot_50) > len(trending_ids):
        cand = random.choice(hot_50)
        if cand.get("product_id") not in trending_ids:
            trending_ids.append(cand.get("product_id"))

    # Gắn scenario tag vào filtered products
    for prod in filtered:
        if prod.get("product_id") in trending_ids:
            prod["scenario"] = "Trending"
        else:
            prod["scenario"] = "Normal"

    return filtered, trending_ids


def generate_daily_sales(product: dict, num_days: int = 30) -> list[dict]:
    """
    Bước 3 (theo Solution.txt): Viết script Python chia số lượng bán (daily_sold).
    
    Với nhóm "Normal":
    - Lấy soldCnt chia đều cho 30 ngày
    - Cộng thêm chút nhiễu ngẫu nhiên (random noise ~20%) để đồ thị đi ngang tự nhiên
    - Tổng 30 ngày vẫn bằng đúng soldCnt
    
    Với nhóm "Trending":
    - Dùng hàm mũ (Exponential function)
    - Lượng bán những ngày đầu tháng rất thấp (1-2 cái)
    - Nhưng những ngày cuối tháng tăng vọt (50-100 cái)
    - Tổng 30 ngày vẫn bằng đúng soldCnt
    """
    total_sold = product.get("soldCnt", 100)
    product_id = product.get("product_id")
    scenario = product.get("scenario", "Normal")

    daily_sales = []
    base_date = datetime.now() - timedelta(days=num_days)

    if scenario == "Trending":
        # Exponential: tập trung ở cuối tháng
        # Công thức: exp(linspace(0, 3, 30)) để tạo đường cong
        exp_curve = np.exp(np.linspace(0, 3, num_days))
        exp_curve = (exp_curve / exp_curve.sum()) * total_sold

        for day in range(num_days):
            # Thêm noise nhỏ để tự nhiên hơn
            daily_amount = max(1, int(exp_curve[day] + random.gauss(0, 1)))
            daily_sales.append(
                {
                    "product_id": product_id,
                    "date": (base_date + timedelta(days=day)).strftime("%Y-%m-%d"),
                    "daily_sold": daily_amount,
                    "scenario": scenario,
                }
            )
    else:
        # Normal: chia đều + noise
        base_daily = total_sold / num_days
        for day in range(num_days):
            # Noise ~20% của giá trị trung bình
            daily_amount = max(1, int(base_daily + random.gauss(0, base_daily * 0.2)))
            daily_sales.append(
                {
                    "product_id": product_id,
                    "date": (base_date + timedelta(days=day)).strftime("%Y-%m-%d"),
                    "daily_sold": daily_amount,
                    "scenario": scenario,
                }
            )

    return daily_sales


def simulate_data() -> None:
    """Main pipeline theo Solution.txt - 3 bước chính"""

    print("=" * 70)
    print("DATA SIMULATION - Theo Solution.txt")
    print("=" * 70)
    print(f"\nScript dir: {SCRIPT_DIR}")
    print(f"Output dir: {OUTPUT_DIR}")

    print("\n[1/3] Loading & validating products...")
    print("      Kiểm tra fields: title, soldCnt, price, image, ratingNumber")
    all_prods = load_and_validate_products()
    if not all_prods:
        return

    print(f"  ✓ Loaded {len(all_prods)} products")

    print("\n[2/3] Filtering 100 products (50 hot + 50 ế) + tagging Trending/Normal...")
    print("      Bước 1: Lọc 50 bán chạy + 50 bán ế")
    print("      Bước 2: Gắn nhãn ~5 Trending + 95 Normal")
    filtered_prods, trending_ids = filter_and_tag_products(all_prods)

    # Save filtered products
    with open(FILTERED_PRODUCTS_JSON, "w", encoding="utf-8") as f:
        json.dump(filtered_prods, f, ensure_ascii=False, indent=2)

    print(f"  ✓ Saved: lazada/output/simulation/filtered_products_100.json")
    print(f"  ✓ Total: {len(filtered_prods)} products")
    print(f"  ✓ Trending: {len(trending_ids)} | Normal: {len(filtered_prods) - len(trending_ids)}")

    if trending_ids:
        print(f"\n  🔥 Trending products:")
        for tid in trending_ids:
            for p in filtered_prods:
                if p.get("product_id") == tid:
                    print(f"    • {p.get('title')[:60]} (sold: {p.get('soldCnt')})")
                    break

    print(f"\n[3/3] Generating sales_history.csv (30 days × 100 products)...")
    print("      Bước 3: Viết script Python chia số lượng bán")
    print("      - Trending: Exponential curve (ngày đầu 1-2 → ngày cuối 50-100)")
    print("      - Normal: Chia đều + noise 20% (đồ thị đi ngang)")
    
    all_daily_sales = []
    for prod in filtered_prods:
        daily_sales = generate_daily_sales(prod)
        all_daily_sales.extend(daily_sales)

    with open(SALES_HISTORY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["product_id", "date", "daily_sold", "scenario"]
        )
        writer.writeheader()
        writer.writerows(all_daily_sales)

    print(f"  ✓ Saved: lazada/output/simulation/sales_history.csv")
    print(f"  ✓ Total rows: {len(all_daily_sales)} (100 products × 30 days)")

    # Summary stats
    trending_rows = [r for r in all_daily_sales if r["scenario"] == "Trending"]
    normal_rows = [r for r in all_daily_sales if r["scenario"] == "Normal"]

    print(f"\n📊 Summary Statistics:")
    print(f"  ├─ Trending rows: {len(trending_rows)}")
    print(f"  │  └─ Pattern: Exponential (1-2 → 50-100 items/day)")
    print(f"  └─ Normal rows: {len(normal_rows)}")
    print(f"     └─ Pattern: Uniform + 20% noise")

    print(f"\n✅ Simulation complete!")
    print(f"\n📁 Output folder structure:")
    print(f"  lazada/")
    print(f"  ├── lazada_products.json (input)")
    print(f"  ├── crawlData.py")
    print(f"  ├── data_simulator.py")
    print(f"  └── output/")
    print(f"      └── simulation/")
    print(f"          ├── filtered_products_100.json")
    print(f"          └── sales_history.csv")
    print("=" * 70)


if __name__ == "__main__":
    simulate_data()

