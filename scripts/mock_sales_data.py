import random
import numpy as np

def generate_exact_daily_sales(total_sold, days=30, scenario="Normal"):
    """
    Hàm phân bổ lượt bán (soldCnt) ra 30 ngày đảm bảo tổng khớp 100%.
    
    Args:
        total_sold (int): Tổng lượt bán từ file JSON.
        days (int): Số ngày cần mô phỏng (mặc định 30).
        scenario (str): "Normal" (Bán đều) hoặc "Trending" (Hàm mũ).
        
    Returns:
        list: Danh sách số lượng bán mỗi ngày (độ dài = days).
    """
    if total_sold <= 0:
        return [0] * days

    if days == 1:
        return [total_sold]

    daily_sales = []

    if scenario == "Normal":
        base_avg = total_sold / days
        for _ in range(days - 1):
            noise_factor = random.uniform(0.7, 1.3)
            sold_today = int(base_avg * noise_factor)
            daily_sales.append(sold_today)
            
    elif scenario == "Trending":
        x_curve = np.linspace(0, 5, days) 
        y_exponential = np.exp(x_curve)
        
        y_normalized = (y_exponential / np.sum(y_exponential)) * total_sold
        
        for i in range(days - 1):
            daily_sales.append(int(y_normalized[i]))
            
    else:
        raise ValueError("Scenario không hợp lệ! Vui lòng chọn 'Normal' hoặc 'Trending'.")

    sum_29_days = sum(daily_sales)
    day_30_sold = total_sold - sum_29_days

    if day_30_sold < 0:
        excess = abs(day_30_sold)
        day_30_sold = 0 
        
        while excess > 0:
            idx = random.randint(0, days - 2)
            if daily_sales[idx] > 0:
                daily_sales[idx] -= 1
                excess -= 1

    daily_sales.append(day_30_sold)
    
    assert sum(daily_sales) == total_sold, f"Lỗi: Tổng chia ({sum(daily_sales)}) không bằng total_sold ({total_sold})"
    
    return daily_sales

if __name__ == "__main__":
    test_sold_cnt = 364  
    print("--- TEST NORMAL SCENARIO ---")
    normal_sales = generate_exact_daily_sales(test_sold_cnt, scenario="Normal")
    print(normal_sales)
    print(f"Tổng Normal: {sum(normal_sales)} (Kỳ vọng: {test_sold_cnt})\n")
    
    print("--- TEST TRENDING SCENARIO ---")
    trend_sales = generate_exact_daily_sales(test_sold_cnt, scenario="Trending")
    print(trend_sales)
    print(f"Tổng Trending: {sum(trend_sales)} (Kỳ vọng: {test_sold_cnt})")