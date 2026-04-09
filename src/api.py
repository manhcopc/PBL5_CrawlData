from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import torch
import requests
from io import BytesIO
from PIL import Image
import os
import uuid

# Các thư viện AI cốt lõi
from pyvi import ViTokenizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from diffusers import StableDiffusionPipeline

# ==========================================
# 0. KHỞI TẠO APP & NẠP MODEL VÀO GPU
# ==========================================
app = FastAPI(
    title="TrendEngine AI Core", 
    description="API Đa phương thức: PhoBERT (Phân tích trend) & Stable Diffusion (Thiết kế)",
    version="1.0.0"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🚀 Đang khởi động hệ thống trên thiết bị: {device.type.upper()}")

# --- TẢI PHOBERT (Đã fine-tune) ---
print(">>> Đang nạp Não Trái (PhoBERT)...")
PHOBERT_PATH = "data/vest/output/models/phobert" # Đổi đường dẫn nếu cần
try:
    nlp_tokenizer = AutoTokenizer.from_pretrained(PHOBERT_PATH)
    nlp_model = AutoModelForSequenceClassification.from_pretrained(PHOBERT_PATH).to(device)
    nlp_model.eval() # Bật chế độ suy luận (chỉ đọc, không train)
    print("Nạp PhoBERT thành công!")
except Exception as e:
    print(f"CẢNH BÁO: Không tìm thấy PhoBERT tại {PHOBERT_PATH}. Chi tiết: {e}")
    nlp_model = None

# --- TẢI STABLE DIFFUSION & IP-ADAPTER ---
print(">>> Đang nạp Não Phải (Stable Diffusion + IP-Adapter)...")
try:
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", 
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
    pipe.set_ip_adapter_scale(0.6) # Cân bằng 60% Form áo gốc - 40% Text mới
    pipe = pipe.to(device)
    print("Nạp Stable Diffusion thành công!")
except Exception as e:
    print(f"CẢNH BÁO: Lỗi tải Stable Diffusion (Có thể tràn RAM). Chi tiết: {e}")
    pipe = None

# ==========================================
# 1. ĐỊNH NGHĨA DATA MODELS (PAYLOAD TỪ BACKEND)
# ==========================================
class ScrapedProduct(BaseModel):
    product_name: str
    image_url: str
    reviews: List[str] # Danh sách text bình luận do BE gửi qua

class TrendRequest(BaseModel):
    request_id: str
    category_keyword: str
    products: List[ScrapedProduct]
    limit: int = 5

class DesignRequest(BaseModel):
    request_id: str
    target_style_prompt: str
    seed_image_urls: List[str]
    num_images: int = 3

# Hàm hỗ trợ: Tải ảnh từ URL
def download_image(url):
    response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")

# ==========================================
# 2. ENDPOINTS CHÍNH (API ROUTES)
# ==========================================

@app.get("/health")
async def health_check():
    """API để Backend kiểm tra xem AI Server có đang rảnh/sống không"""
    return {
        "status": "online",
        "device": str(device),
        "modules": {
            "phobert_ready": nlp_model is not None,
            "stable_diffusion_ready": pipe is not None
        }
    }

@app.post("/api/v1/analyze-trend")
async def analyze_trend(payload: TrendRequest):
    """BƯỚC 1: Nhận data cào từ BE -> Chấm điểm Khen/Chê -> Trả về Top Trends"""
    if nlp_model is None:
        raise HTTPException(status_code=500, detail="PhoBERT module is offline.")
    
    print(f"\n🔍 [PhoBERT] Đang phân tích {len(payload.products)} sản phẩm cho Request: {payload.request_id}")
    analyzed_products = []
    
    try:
        for item in payload.products:
            if not item.reviews:
                continue # Bỏ qua sản phẩm không có ai bình luận
                
            # Tokenize toàn bộ bình luận bằng PyVi
            processed_texts = [ViTokenizer.tokenize(text) for text in item.reviews]
            inputs = nlp_tokenizer(processed_texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            
            # Chạy Inference tính điểm
            with torch.no_grad():
                outputs = nlp_model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                # Lấy cột index 1 (Giả định nhãn 1 là Positive/Khen)
                positive_scores = probs[:, 1].cpu().tolist() 
                
            # Tính % Khen trung bình của cái áo này
            avg_positive_rate = sum(positive_scores) / len(positive_scores)
            
            analyzed_products.append({
                "product_name": item.product_name,
                "source_image_url": item.image_url,
                "positive_rate": round(avg_positive_rate, 2),
                "total_reviews": len(item.reviews)
            })
            
        # Sắp xếp giảm dần theo điểm Khen và lấy Top
        analyzed_products.sort(key=lambda x: x["positive_rate"], reverse=True)
        top_trends = analyzed_products[:payload.limit]
        
        print(f"✅ Đã tìm ra {len(top_trends)} siêu trend!")
        return {
            "status": "success",
            "request_id": payload.request_id,
            "trends": top_trends
        }
    
    except Exception as e:
        print(f"❌ Lỗi PhoBERT: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/generate-design")
async def generate_design(payload: DesignRequest):
    """BƯỚC 2: Nhận Link ảnh gốc + Text Style -> Vẽ áo mới -> Trả link ảnh"""
    if pipe is None:
        raise HTTPException(status_code=500, detail="Stable Diffusion module is offline.")
    
    print(f"\n🎨 [SD] Đang vẽ thiết kế cho Request: {payload.request_id}")
    print(f"   - Style: {payload.target_style_prompt}")
    
    try:
        # 1. Tải ảnh gốc để làm Hạt giống (Seed) cho IP-Adapter
        print("   - Đang tải ảnh gốc từ hệ thống...")
        trend_images = [download_image(url) for url in payload.seed_image_urls]
        
        # 2. Xây dựng bộ Prompt hoàn chỉnh (Ép chất lượng cao)
        base_prompt = "High-end fashion photography, a modern premium vest suit, sharp tailoring, cinematic lighting, 8k resolution, highly detailed, studio background"
        full_prompt = f"{payload.target_style_prompt}, {base_prompt}"
        negative_prompt = "low quality, blurry, distorted, messy, extra limbs, bad anatomy, ugly, deformed, text, watermark, messy background"

        generated_urls = []
        os.makedirs("./outputs", exist_ok=True) # Thư mục lưu tạm
        
        # 3. Kích hoạt Stable Diffusion vẽ ảnh
        for i in range(payload.num_images):
            print(f"   - Đang render bức ảnh {i+1}/{payload.num_images}...")
            # Dùng autocast float16 để tăng tốc và tiết kiệm VRAM
            with torch.autocast("cuda"):
                image = pipe(
                    prompt=full_prompt,
                    negative_prompt=negative_prompt,
                    ip_adapter_image=[trend_images], # Đóng gói thành mảng 2 chiều cho nhiều ảnh
                    num_inference_steps=30,
                    guidance_scale=7.5
                ).images[0]
            
            # 4. Lưu ảnh & Lấy Link
            filename = f"design_{payload.request_id}_{uuid.uuid4().hex[:6]}.png"
            local_path = f"./outputs/{filename}"
            image.save(local_path)
            
            # TODO: Ở môi trường thật, m viết code upload file này lên Cloudinary/S3
            # Tạm thời giả lập trả về URL tĩnh
            mock_cloud_url = f"https://your-domain.com/outputs/{filename}"
            generated_urls.append(mock_cloud_url)

        print("✅ Hoàn tất thiết kế!")
        return {
            "status": "success",
            "request_id": payload.request_id,
            "generated_designs": generated_urls
        }

    except Exception as e:
        print(f"❌ Lỗi Stable Diffusion: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# 3. LỆNH KHỞI ĐỘNG SERVER (Chỉ dùng khi test local)
# ==========================================
if __name__ == "__main__":
    import uvicorn
    # Khi code trên mạng/Colab, nên dùng host="0.0.0.0"
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)