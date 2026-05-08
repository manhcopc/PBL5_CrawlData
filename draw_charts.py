import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# CẤU HÌNH GIAO DIỆN CHUẨN BÁO CÁO HỌC THUẬT
# ==========================================
# Dùng theme whitegrid cho sáng sủa, context 'talk' để font to rõ khi in poster
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams['font.family'] = 'sans-serif'

# Đọc file dữ liệu
try:
    df = pd.read_csv('metrics_log.csv')
except FileNotFoundError:
    print("❌ Không tìm thấy file metrics_log.csv. Hãy chạy API vài lần để tạo log trước!")
    exit()

# ==========================================
# 1. BIỂU ĐỒ VRAM (GEN AI) - DẠNG LINE & AREA
# ==========================================
gen_df = df[df['api_name'] == 'generate_design'].reset_index()

if not gen_df.empty:
    plt.figure(figsize=(10, 6))
    
    # Vẽ line chart điểm đánh dấu (marker) lớn, màu đỏ mận (alizarin)
    ax1 = sns.lineplot(
        data=gen_df, x=gen_df.index, y='peak_vram_gb',
        marker='o', markersize=10, color='#e74c3c', linewidth=3
    )
    
    # Kỹ thuật ăn tiền: Tô màu gradient mờ dưới đường line
    plt.fill_between(gen_df.index, gen_df['peak_vram_gb'], color='#e74c3c', alpha=0.1)

    # Tiêu đề và nhãn
    plt.title('ĐỈNH TIÊU THỤ VRAM QUA CÁC LẦN SINH ẢNH (GENAI)', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Lần gọi API (Request Index)', fontsize=13)
    plt.ylabel('VRAM Tiêu thụ (GB)', fontsize=13)
    plt.ylim(0, 16) # Trục Y cố định 16GB để thấy VRAM luôn an toàn

    # Hiển thị số liệu trực tiếp lên từng điểm (chỉ lấy 2 số thập phân)
    for x, y in zip(gen_df.index, gen_df['peak_vram_gb']):
        plt.text(x, y + 0.6, f'{y:.2f}GB', ha='center', va='bottom', 
                 fontsize=11, color='#c0392b', fontweight='bold')

    plt.tight_layout()
    plt.savefig('vram_chart_seaborn.png', dpi=300, transparent=False)
    print("✅ Đã lưu vram_chart_seaborn.png (Bản đẹp)")
    plt.clf() # Xóa canvas để vẽ hình tiếp theo

# ==========================================
# 2. BIỂU ĐỒ THỜI GIAN NLP (PHOBERT) - DẠNG BAR
# ==========================================
nlp_df = df[df['api_name'] == 'analyze_trend'].reset_index()

if not nlp_df.empty:
    plt.figure(figsize=(10, 6))
    
    # Dùng barplot với màu xanh lam nhạt (peter river), bo viền đen mỏng
    ax2 = sns.barplot(
        data=nlp_df, x=nlp_df.index, y='latency_seconds',
        color='#3498db', edgecolor='#2c3e50', linewidth=1.5
    )

    # Tiêu đề và nhãn
    plt.title('THỜI GIAN XỬ LÝ PHÂN TÍCH XU HƯỚNG (PHOBERT)', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Lần gọi API (Request Index)', fontsize=13)
    plt.ylabel('Thời gian phản hồi (Giây)', fontsize=13)

    # Thêm số liệu trực tiếp lên đỉnh các cột
    for i in ax2.containers:
        ax2.bar_label(i, fmt='%.3fs', padding=3, fontsize=12, fontweight='bold', color='#2980b9')

    plt.tight_layout()
    plt.savefig('nlp_latency_chart_seaborn.png', dpi=300, transparent=False)
    print("✅ Đã lưu nlp_latency_chart_seaborn.png (Bản đẹp)")