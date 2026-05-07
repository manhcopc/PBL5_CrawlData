import os
import json
import torch
import requests
from PIL import Image
from io import BytesIO
from transformers import CLIPProcessor, CLIPModel
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN
# ==========================================
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_TREND_PATH = os.path.join(ROOT_DIR, "data", "vest", "output", "simulation", "trending_vests_01.json")

# Tạo thư mục mới để chứa Features (Vector)
FEATURE_DIR = os.path.join(ROOT_DIR, "data", "vest", "output", "features")
os.makedirs(FEATURE_DIR, exist_ok=True)
OUTPUT_FEATURE_PATH = os.path.join(FEATURE_DIR, "trending_features_01.pt")

# Chọn thiết bị chạy (GPU nếu có, không thì CPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def download_image(url):
    """Hàm tải ảnh từ mạng, giả dạng trình duyệt để tránh bị Shopee/Lazada chặn"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        return image
    except Exception as e:
        print(f"Lỗi tải ảnh: {e}")
        return None

def main():
    if not os.path.exists(JSON_TREND_PATH):
        print(f"Không tìm thấy file: {JSON_TREND_PATH}")
        return

    # 1. Khởi động AI CLIP
    print(f">>> Đang đánh thức AI CLIP (openai/clip-vit-base-patch32) trên {DEVICE}...")
    # Tải 2 thứ: Processor (để xử lý ảnh đầu vào) và Model (để xuất vector)
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval() # Bật chế độ suy luận (không train)

    # 2. Đọc file Trend
    print(f">>> Đang nạp danh sách Siêu Trend từ {os.path.basename(JSON_TREND_PATH)}...")
    with open(JSON_TREND_PATH, 'r', encoding='utf-8') as f:
        trending_products = json.load(f)
        
    if not trending_products:
        print("File trend trống không. Hãy kiểm tra lại bước Lọc Trend.")
        return

    features_dict = {}
    total = len(trending_products)
    
    print("\nBẮT ĐẦU CHO CLIP 'NHÌN' VÀ TRÍCH XUẤT ĐẶC TRƯNG...")
    
    with torch.no_grad(): # Tắt tính toán đạo hàm cho nhẹ RAM
        for idx, item in enumerate(trending_products, 1):
            prod_id = item['product_id']
            img_url = item['image_url']
            
            print(f"[{idx}/{total}] Đang xử lý ID: {prod_id}...")
            
            # Tải ảnh
            image = download_image(img_url)
            if image is None:
                continue
                
            # Đưa ảnh qua Processor để resize, normalize chuẩn form của CLIP
            inputs = processor(images=image, return_tensors="pt").to(DEVICE)
            
            # Đưa vào Model để rút trích Vector
            image_features = model.get_image_features(**inputs)
            
            # (Tùy chọn cực xịn) Chuẩn hóa Vector (Normalize) để sau này dễ tính toán khoảng cách
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            
            # Lưu cái Vector (chỉ lấy số, đẩy về CPU để tránh tốn VRAM GPU)
            features_dict[prod_id] = image_features.cpu()
            print(f"Đã trích xuất thành công Vector (size: {image_features.shape})")

    # 3. Xuất file kết quả
    print("\n" + "="*60)
    print(f"TỔNG KẾT: Đã trích xuất đặc trưng cho {len(features_dict)}/{total} sản phẩm Trend.")
    
    # Lưu toàn bộ dict này thành file PyTorch (.pt)
    torch.save(features_dict, OUTPUT_FEATURE_PATH)
    
    print(f"Các 'Hạt giống Trend' đã được nén thành file Toán học tại: {OUTPUT_FEATURE_PATH}")
    print("GIAI ĐOẠN 2 HOÀN TẤT. SẴN SÀNG GỌI STABLE DIFFUSION!")
    print("="*60)

if __name__ == "__main__":
    main()