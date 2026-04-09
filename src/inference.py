import os
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from pyvi import ViTokenizer
import traceback

current_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(current_dir)
MODEL_PATH = os.path.join(ROOT_DIR, "output", "models", "phobert_genz_v1.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PhoBertSentimentClassifier(torch.nn.Module):
    def __init__(self, n_classes=2):
        super(PhoBertSentimentClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained("wonrax/phobert-base-vietnamese-sentiment")
        self.drop = torch.nn.Dropout(p=0.3)
        self.out = torch.nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.out(self.drop(outputs.pooler_output))

class TrendEnginePredictor:
    def __init__(self, model_path):
        print(f"Dang khoi dong PhoBERT tren {DEVICE}...")
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
        self.model = PhoBertSentimentClassifier(n_classes=2)
        
        print(f"Dang nap trong so tu: {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Khong tim thay file model tai: {model_path}")
        
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True), strict=False)
        self.model.to(DEVICE)
        self.model.eval()
        print("Da load Model thanh cong!")

    def predict(self, text):
        processed_text = ViTokenizer.tokenize(text)
        encoding = self.tokenizer(
            processed_text, truncation=True, add_special_tokens=True, max_length=128,
            padding='max_length', return_attention_mask=True, return_tensors='pt',
        )
        input_ids = encoding['input_ids'].to(DEVICE)
        attention_mask = encoding['attention_mask'].to(DEVICE)
        
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            probs = F.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probs, dim=1)
            
        labels = {0: "Tieu cuc", 1: "Tich cuc"}
        return labels[predicted_class.item()], confidence.item() * 100

def main():
    try:
        predictor = TrendEnginePredictor(MODEL_PATH)
    except Exception as e:
        print("\nLoi he thong:")
        traceback.print_exc()
        return
        
    print("\n" + "="*50)
    print("AI CHATBOT: TEST CAM XUC GEN Z")
    print("Go 'exit', 'quit' hoac 'q' de thoat.")
    print("="*50)
    
    while True:
        text = input("\nNhap comment ban muon test: ")
        if text.lower() in ['exit', 'quit', 'q']:
            print("Da thoat!")
            break
        if not text.strip():
            continue
            
        sentiment, score = predictor.predict(text)
        print(f"Ket qua AI: {sentiment} (Do tu tin: {score:.2f}%)")

if __name__ == "__main__":
    main()