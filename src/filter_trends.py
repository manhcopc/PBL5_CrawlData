import pandas as pd
import json
import os
from transformers import pipeline
from pyvi import ViTokenizer
import warnings

warnings.filterwarnings("ignore")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(ROOT_DIR, "data", "vest", "output", "simulation", "merged_vest_data.csv")
OUTPUT_TREND_PATH = os.path.join(ROOT_DIR, "data", "vest", "output", "simulation", "trending_vests.json")

TREND_THRESHOLD = 0.7

def main():
    if not os.path.exists(CSV_PATH):
        print(f"Không tìm thấy file: {CSV_PATH}")
        print("Nhớ chạy merge_data cho folder vest trước nhé!")
        return

    print(">>> Đang tải Model PhoBERT (wonrax)... (Lần đầu sẽ mất xíu thời gian)")
    analyzer = pipeline("sentiment-analysis", model="wonrax/phobert-base-vietnamese-sentiment")
    
    print(f">>> Đang đọc dữ liệu từ: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    
    # Ép kiểu product_id về chuỗi cho chắc cú
    df['product_id'] = df['product_id'].astype(str)
    
    if 'scenario' in df.columns:
        print("\nVÒNG 1: Đang lọc các sản phẩm có lượt bán 'Trending'...")
        df_trending_sales = df[df['scenario'] == 'Trending']
        print(f"Đã giữ lại {len(df_trending_sales)} bình luận thuộc các sản phẩm bán chạy.")
    else:
        print("Không tìm thấy cột 'scenario', sẽ quét toàn bộ dữ liệu.")
        df_trending_sales = df
    
    if df_trending_sales.empty:
        print("Không có sản phẩm nào đang Trending về lượt bán. Dừng chương trình.")
        return

    grouped = df_trending_sales.groupby(['product_id', 'title', 'image'])
    
    trending_products = []
    total_products = len(grouped)
    current = 0
    
    print("\n🕵️VÒNG 2: BẮT ĐẦU QUÉT CẢM XÚC BẰNG AI...")
    for (prod_id, title, img_url), group in grouped:
        current += 1
        pos_count = 0
        total_reviews = len(group)
        
        # Đưa từng comment cho PhoBERT chấm
        for review in group['review_text']:
            # Xử lý trường hợp review bị null (NaN)
            if pd.isna(review):
                continue
                
            tokenized = ViTokenizer.tokenize(str(review))
            result = analyzer(tokenized)[0]
            
            # Cứ thấy nhãn POS là cộng 1 điểm khen
            if result['label'] == 'POS':
                pos_count += 1
                
        # Tính tỷ lệ khen ngợi
        positive_rate = pos_count / total_reviews if total_reviews > 0 else 0
        
        status = "ĐẠT" if positive_rate >= TREND_THRESHOLD else "LOẠI"
        print(f"[{current}/{total_products}] ID: {prod_id} | Khen: {pos_count}/{total_reviews} ({(positive_rate*100):.0f}%) | {status}")
        
        # Nếu tỷ lệ khen vượt mức tiêu chuẩn -> Đưa vào danh sách Trend
        if positive_rate >= TREND_THRESHOLD:
            trending_products.append({
                "product_id": prod_id,
                "title": title,
                "image_url": img_url,
                "positive_rate": round(positive_rate, 2),
                "total_reviews": total_reviews
            })

    # Xuất file kết quả cho CLIP
    print("\n" + "="*60)
    print(f"TỔNG KẾT: Đã lọc được {len(trending_products)}/{total_products} bộ vest lọt vào danh sách 'SIÊU TREND'!")
    
    with open(OUTPUT_TREND_PATH, 'w', encoding='utf-8') as f:
        json.dump(trending_products, f, ensure_ascii=False, indent=4)
        
    print(f"Đã lưu danh sách Trend (chứa Link Ảnh) tại: {OUTPUT_TREND_PATH}")
    print("GIAI ĐOẠN 1 HOÀN TẤT. SẴN SÀNG CHUYỂN ẢNH CHO CLIP!")
    print("="*60)

if __name__ == "__main__":
    main()