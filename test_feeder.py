import pandas as pd
import requests
import time

# Configuration
# Replace with your actual ngrok URL running on Google Colab
AI_SERVER_URL = "https://0baa-34-125-130-67.ngrok-free.app/api/v1/analyze-trend"
CSV_FILE_PATH = 'data/vest/output/simulation/simulated_reviews.csv'

print("Starting Data Feeder Simulation...")

# 1. Read simulated reviews from CSV
try:
    df = pd.read_csv(CSV_FILE_PATH, names=['product_id', 'review_text', 'label'])
except FileNotFoundError:
    print(f"Error: Could not find the file at {CSV_FILE_PATH}")
    exit()

# Group reviews by product ID
grouped_reviews = df.groupby('product_id')['review_text'].apply(list).to_dict()

# 2. Package the payload for Analyze Trend API
products_payload = []
for pid, reviews in grouped_reviews.items():
    products_payload.append({
        "product_name": f"Vest Model {pid}",
        # Appending resize parameters to prevent OOM errors on the AI Server
        "image_url": "https://images.unsplash.com/photo-1591047139829-d91aecb6caea?w=512&h=512&fit=crop", 
        "reviews": reviews,
        "scenario": "Normal",
        "sales_velocity": 50,
        "created_at": "2026-05-01T10:00:00Z"
    })

payload = {
    "request_id": "simulated_batch_001",
    "category_keyword": "vest",
    "products": products_payload,
    "limit": 5
}

# 3. Send data to AI Server for trend analysis
print(f"Sending {len(products_payload)} products to AI Server for trend analysis...")
try:
    response = requests.post(AI_SERVER_URL, json=payload, timeout=30)
except requests.exceptions.RequestException as e:
    print(f"Network error during trend analysis: {e}")
    exit()

if response.status_code == 200:
    print("Analysis successful. Results:")
    results = response.json().get('trends', [])
    
    for item in results:
        print(f"\n- Product: {item['product_name']}")
        print(f"  Trend Score: {item['trend_score']}")
        print(f"  Extracted Style Keywords: {item['style_keywords']}")

    print("\n" + "="*60)
    print("STEP 2: HANDOVER TO GENERATIVE AI")
    print("="*60)
    
    # 4. Extract Top 1 Trend for generative design
    if not results:
        print("No trends found to generate design.")
        exit()
        
    top_1_product = results[0]
    top_1_keywords = ", ".join(top_1_product['style_keywords'])
    base_image = top_1_product['source_image_url']
    
    print(f"Using Top 1 product as inspiration: {top_1_product['product_name']}")
    print(f"Design prompt based on keywords: {top_1_keywords}")

    gen_payload = {
        "request_id": "gen_auto_001",
        "target_style_prompt": f"high fashion blazer vest, {top_1_keywords} texture, luxury cinematic lighting, 8k resolution, photorealistic",
        "base_image_url": base_image,
        "num_images": 1,
        "canny_low_threshold": 100,
        "canny_high_threshold": 200
    }

    GEN_API_URL = AI_SERVER_URL.replace("/analyze-trend", "/generate-design")
    
    # 5. Send request to Generative AI
    print("Submitting generation request to server...")
    try:
        gen_response = requests.post(GEN_API_URL, json=gen_payload, timeout=30)
    except requests.exceptions.RequestException as e:
        print(f"Network error during generation request: {e}")
        exit()
        
    if gen_response.status_code == 202:
        job_id = gen_response.json().get('job_id')
        print(f"Job accepted. Job ID: {job_id}")
        
        # 6. Polling the server for job status
        JOB_API_URL = GEN_API_URL.replace("/generate-design", f"/generation-jobs/{job_id}")
        
        while True:
            try:
                job_status_response = requests.get(JOB_API_URL, timeout=10)
                if job_status_response.status_code == 200:
                    job_status = job_status_response.json()
                    status = job_status.get("status")
                    
                    if status == "success":
                        print("\nGENERATION COMPLETE. VIEW RESULT AT:")
                        print(job_status["result"]["images"][0]["url"])
                        break
                    elif status == "failed":
                        print(f"\nGeneration failed: {job_status.get('error')}")
                        break
                    else:
                        print(f"   ... Server is working (status: {status}). Waiting 5 seconds...")
                        time.sleep(5)
                else:
                    print(f"   ... Unexpected status code {job_status_response.status_code}. Retrying in 5 seconds...")
                    time.sleep(5)
            
            except requests.exceptions.RequestException as e:
                print(f"   ... Network instability detected ({e}). Retrying in 5 seconds...")
                time.sleep(5)
    else:
        print(f"Error submitting generation request: {gen_response.status_code} - {gen_response.text}")
else:
    print(f"Error during trend analysis: {response.status_code} - {response.text}")