import pandas as pd
import matplotlib.pyplot as plt

# Đọc file dữ liệu
df = pd.read_csv('metrics_log.csv')

# Lọc riêng data của GenAI
gen_df = df[df['api_name'] == 'generate_design'].reset_index()

# 1. Vẽ biểu đồ VRAM (Line chart) để chứng minh hệ thống không bị tràn RAM
plt.figure(figsize=(10, 5))
plt.plot(gen_df.index, gen_df['peak_vram_gb'], marker='o', color='red', linestyle='-')
plt.title('Đỉnh tiêu thụ VRAM qua các lần sinh ảnh (Stable Diffusion)')
plt.xlabel('Lần gọi API (Request Index)')
plt.ylabel('VRAM Tiêu thụ (GB)')
plt.ylim(0, 16) # Giả sử dùng T4 16GB
plt.grid(True)
plt.savefig('vram_chart.png', dpi=300) # Xuất ảnh nét căng >150 dpi chuẩn Bách Khoa
print("Đã lưu vram_chart.png")

# Lọc riêng data của NLP
nlp_df = df[df['api_name'] == 'analyze_trend']

# 2. Vẽ biểu đồ thời gian phản hồi NLP (Bar chart)
plt.figure(figsize=(8, 5))
plt.bar(range(len(nlp_df)), nlp_df['latency_seconds'], color='blue')
plt.title('Thời gian xử lý Phân tích Xu hướng (PhoBERT)')
plt.xlabel('Lần gọi API')
plt.ylabel('Thời gian (Giây)')
plt.grid(axis='y')
plt.savefig('nlp_latency_chart.png', dpi=300)
print("Đã lưu nlp_latency_chart.png")