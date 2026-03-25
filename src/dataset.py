import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from pyvi import ViTokenizer 

class GenZReviewDataset(Dataset):
    def __init__(self, csv_path, model_name="vinai/phobert-base", max_len=128):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_len = max_len
        
        self.label_dict = {'positive': 1, 'negative': 0}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        review = str(self.df.iloc[item]['review_text'])
        sentiment = self.df.iloc[item]['sentiment']
    
        review = ViTokenizer.tokenize(review)
        
        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'review_text': review,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.label_dict[sentiment], dtype=torch.long)
        }

if __name__ == "__main__":
    DATA_PATH = "output/simulation/simulated_reviews.csv"
    try:
        ds = GenZReviewDataset(DATA_PATH)
        print(f">>> Đã load thành công {len(ds)} mẫu dữ liệu.")
        sample = ds[0]
        print(f"Câu đầu tiên: {sample['review_text']}")
        print(f"Nhãn (Label): {sample['labels']}")
    except Exception as e:
        print(f"Lỗi: {e}. Kiểm tra lại đường dẫn file CSV nhé m!")