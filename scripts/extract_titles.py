import json
import os

def main():
    print(">>> ĐANG TRÍCH XUẤT DANH SÁCH SẢN PHẨM <<<")
    
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(SCRIPT_DIR)
    
    json_path = os.path.join(ROOT_DIR, 'filtered_products_100.json')
    if not os.path.exists(json_path):
        json_path = os.path.join(ROOT_DIR, 'data', 'vest', 'output', 'simulation', 'filtered_products_100.json')
        
    output_txt_path = os.path.join(ROOT_DIR, 'data', 'vest', 'output','filter', 'product_title.txt')

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            products = json.load(f)
    except FileNotFoundError:
        print(f"Không tìm thấy file JSON tại: {json_path}")
        return

    with open(output_txt_path, 'w', encoding='utf-8') as f:
        for i, p in enumerate(products):
            clean_title = str(p['title']).replace('\n', ' ').strip()
            line = f"ID: {p['product_id']} | Title: {clean_title}\n"
            f.write(line)

    print(f"\nXONG! Đã trích xuất {len(products)} sản phẩm.")

if __name__ == "__main__":
    main()