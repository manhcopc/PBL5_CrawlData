import os
import json
import time
import random 
import pandas as pd
from google import genai 
from google.genai import types 
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("LỖI: Chưa tìm thấy GOOGLE_API_KEY! Hãy chắc chắn bạn đã tạo file .env")

client = genai.Client(api_key=GOOGLE_API_KEY)

def generate_reviews_for_product(product_title, product_id, retries=3):
    # Random tỉ lệ khen chê (Luôn là 10 comment nhưng xê dịch từ 6-10 khen)
    pos_count = random.randint(6, 10)
    neg_count = 10 - pos_count
    
    # Persona Injection (Bơm nhân cách)
    vibes = [
        "hào hứng, năng động, hay dùng teencode và tiếng lóng mạng xã hội (slay, keo lỳ, chấn bé đù, xịn xò).",
        "lười biếng, viết cực kỳ ngắn gọn, không viết hoa chữ cái đầu, thỉnh thoảng sai chính tả nhẹ (vd: dep, vai mat, okla).",
        "khó tính, hay soi từng đường kim mũi chỉ, so sánh với ảnh mẫu.",
        "cảm xúc mạnh, lạm dụng dấu chấm than và kéo dài chữ (vd: đẹpppppp quá trời ơiiii!!!, xuất sắccccc).",
        "thực tế, hay review chi tiết về việc mặc đi đâu (vd: mặc đi cafe chụp ảnh bao cháy, mặc đi học thoải mái)."
    ]
    current_vibe = random.choice(vibes)

    prompt = f"""
    Bạn là một tập hợp các khách hàng thực tế vừa nhận được sản phẩm thời trang. 
    Tên sản phẩm: "{product_title}". 
    - QUY TẮC SỐNG CÒN: Hãy tự phân tích Tên sản phẩm để xác định chính xác ĐÂY LÀ ÁO, QUẦN, VÁY, GIÀY hay PHỤ KIỆN. Tuyệt đối không gọi sai tên loại trang phục (ví dụ tên là Áo mà comment khen Quần là bị phạt). Nếu tên sản phẩm không rõ ràng, hãy dùng từ chung chung như "set đồ", "đồ", "bộ này".
    Nhiệm vụ: Tạo 10 bình luận review sản phẩm. 
    Yêu cầu ĐẶC BIỆT về cảm xúc và văn phong (HÃY ÁP DỤNG TRIỆT ĐỂ):
    - Mang đậm phong cách: {current_vibe}
    - KHÔNG được rập khuôn. Mỗi bình luận phải có độ dài ngắn khác nhau (từ 3 chữ đến 36 chữ).
    - Hạn chế bắt đầu bằng "Sản phẩm", "Quần này", "Áo này". Hãy bắt đầu tự nhiên như người thật đang nói chuyện.
    - Cố tình viết chữ thường đầu câu, thêm emoji (🔥, 😍, 😭, 😡) một cách tự nhiên.

    Yêu cầu về phân bổ:
    - Đúng {pos_count} bình luận Tích cực (positive): Phải khéo léo chèn các từ khóa thời trang Gen Z (form rộng, hack dáng, y2k, chất liệu, màu pastel, vintage, streetwear...).
    - Đúng {neg_count} bình luận Tiêu cực/Phàn nàn (negative): Chê về giao hàng, đóng gói, chỉ thừa, nhăn, lệch màu, hoặc form không như kỳ vọng. (Nếu số này là 0 thì bỏ qua).

    Cấu trúc JSON BẮT BUỘC TRẢ VỀ:
    [
        {{"product_id": "{product_id}", "review_text": "text comment 1", "sentiment": "positive"}},
        {{"product_id": "{product_id}", "review_text": "text comment 2", "sentiment": "negative"}}
    ]
    """

    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model='gemini-2.0-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.9, 
                ),
            )
            
            raw_text = response.text.strip()
            reviews_list = json.loads(raw_text)
            
            if isinstance(reviews_list, list):
                return reviews_list
            else:
                raise ValueError("Model không trả về một List JSON hợp lệ.")
            
        except json.JSONDecodeError:
            print(f"[Lỗi] Trả về không phải JSON cho SP {product_id}. Thử lại lần {attempt + 1}...")
            time.sleep(2)
        except Exception as e:
            error_msg = str(e).lower()
            if "429" in error_msg or "quota" in error_msg or "exhausted" in error_msg:
                sleep_time = 60 * (attempt + 1)
                print(f"[Cảnh báo] Quá tải API cho SP {product_id}. Đang ngủ đông {sleep_time}s để vượt ải...")
                time.sleep(sleep_time)
            else:
                print(f"[Lỗi server/mạng] {e} ở SP {product_id}. Thử lại lần {attempt + 1}...")
                time.sleep(5)
                
    return []

def main():
    print(">>> BẮT ĐẦU CHẠY MÔ PHỎNG REVIEWS (BẢN GEMINI 2.0 FLASH) <<<")
    
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(SCRIPT_DIR)
    
    input_file_path = os.path.join(ROOT_DIR, 'output', 'simulation', 'filtered_products_100.json')
    output_file_path = os.path.join(ROOT_DIR, 'output', 'simulation', 'simulated_reviews.csv')
    
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            products = json.load(f)
    except FileNotFoundError:
        print(f"LỖI TÌM FILE: Không thấy file tại '{input_file_path}'.")
        return
        
    all_simulated_reviews = []
    
    processed_ids = set()
    if os.path.exists(output_file_path):
        try:
            df_existing = pd.read_csv(output_file_path)
            processed_ids = set(df_existing['product_id'].astype(str))
            print(f"[*] Tìm thấy {len(processed_ids)} sản phẩm đã xử lý. Sẽ bỏ qua các sản phẩm này.")
        except Exception as e:
            print(f"[*] Không thể đọc file CSV cũ, sẽ tạo mới. Lỗi: {e}")
            pd.DataFrame(columns=["product_id", "review_text", "sentiment"]).to_csv(output_file_path, index=False, encoding='utf-8-sig')
    else:
        pd.DataFrame(columns=["product_id", "review_text", "sentiment"]).to_csv(output_file_path, index=False, encoding='utf-8-sig')
    
    for i, product in enumerate(products):
        prod_id = str(product['product_id'])
        
        if prod_id in processed_ids:
            print(f"[{i+1}/{len(products)}] Bỏ qua: {product['title'][:30]}... (Đã có)")
            continue
            
        print(f"[{i+1}/{len(products)}] Đang sinh comment cho: {product['title'][:30]}...")
        
        reviews = generate_reviews_for_product(product['title'], prod_id)
        
        if reviews:
            all_simulated_reviews.extend(reviews)
            df_temp = pd.DataFrame(reviews)
            df_temp.to_csv(output_file_path, mode='a', header=False, index=False, encoding='utf-8-sig')

        time.sleep(12) 

    if all_simulated_reviews:
        print(f"\nXONG! Đã sinh mới thành công {len(all_simulated_reviews)} bình luận đầy cảm xúc.")
    else:
        print("\nQuá trình hoàn tất (Không có dữ liệu mới nào cần sinh thêm).")

if __name__ == "__main__":
    main()