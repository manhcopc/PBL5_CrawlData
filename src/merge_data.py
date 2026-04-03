import pandas as pd
import json
import os

def main():
    print(">>> Đang khởi động tiến trình Merge Data...")

    # Cấu hình đường dẫn trỏ thẳng vào folder data/vest
    current_dir = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(current_dir)
    
    json_path = os.path.join(ROOT_DIR, "data", "vest", "output", "simulation", "filtered_products_100.json") 
    csv_path = os.path.join(ROOT_DIR, "data", "vest", "output", "simulation", "simulated_reviews.csv")
    output_path = os.path.join(ROOT_DIR, "data", "vest", "output", "simulation", "merged_vest_data.csv")

    if not os.path.exists(json_path):
        print(f"Lỗi: Không tìm thấy file JSON tại: {json_path}")
        return
    if not os.path.exists(csv_path):
        print(f"Lỗi: Không tìm thấy file CSV tại: {csv_path}")
        return

    print(">>> Đang đọc file reviews...")
    with open(csv_path, 'r', encoding='utf-8') as f:
        first_line = f.readline()
        
    if 'product_id' in first_line:
        df_reviews = pd.read_csv(csv_path)
    else:
        df_reviews = pd.read_csv(csv_path, header=None, names=['product_id', 'review_text', 'sentiment'])
        
    df_reviews['product_id'] = df_reviews['product_id'].astype(str) 

    print(">>> Đang đọc file JSON sản phẩm...")
    with open(json_path, 'r', encoding='utf-8') as f:
        products_data = json.load(f)
    
    df_products = pd.DataFrame(products_data)
    df_products['product_id'] = df_products['product_id'].astype(str)
    
    print(">>> Đang tiến hành ghép nối (Merge TOÀN BỘ DATA)...")
    # Ghép nối bằng Inner Join thay vì Left Join để chắc chắn sản phẩm nào có review mới lấy
    df_merged = pd.merge(df_reviews, df_products, on='product_id', how='inner')

    # Định nghĩa các cột mong muốn (lấy cột nào có thì lấy)
    desired_cols = [
        'product_id', 'title', 'review_text', 'sentiment', 
        'image', 'price', 'soldCnt', 'ratingNumber', 
        'rating_score', 'shop_name', 'product_url', 'scenario'
    ]
    
    existing_cols = [col for col in desired_cols if col in df_merged.columns]
    df_merged = df_merged[existing_cols]

    df_merged.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print("\n" + "-" * 40)
    print(f"ĐÃ MERGE XONG!")
    print(f"Dữ liệu tổng hợp lưu tại: {output_path}")
    print(f"Tổng số dòng dữ liệu (Reviews): {len(df_merged)}")
    print(f"Các trường dữ liệu: {', '.join(existing_cols)}")
    print("-" * 40)

if __name__ == "__main__":
    main()