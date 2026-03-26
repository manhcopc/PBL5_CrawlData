import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW 
from transformers import get_linear_schedule_with_warmup
import pandas as pd
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from dataset import GenZReviewDataset
from nlp_processor import PhoBertSentimentClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 5
BATCH_SIZE = 16
LEARNING_RATE = 2e-5

def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0
    
    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["labels"].to(device)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, labels)

        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        # Backward pass
        loss.backward()
        # Clip gradient để tránh bùng nổ (Gradient Explosion)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, sum(losses) / len(losses)

def main():
    print(f"\n>>> THIẾT BỊ ĐANG SỬ DỤNG: {DEVICE}")
    
    # 1. Khởi tạo đường dẫn
    root_dir = os.path.dirname(current_dir)
    csv_path = os.path.join(root_dir, "output", "simulation", "simulated_reviews.csv")
    model_save_dir = os.path.join(root_dir, "output", "models")
    os.makedirs(model_save_dir, exist_ok=True)
    
    if not os.path.exists(csv_path):
        print(f"LỖI: Không tìm thấy file dữ liệu tại: {csv_path}")
        return

    # 2. Chuẩn bị Data
    print(">>> Đang nạp dữ liệu Gen")
    dataset = GenZReviewDataset(csv_path)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 3. Khởi tạo Model & Optimizer
    model = PhoBertSentimentClassifier(n_classes=2)
    
    # Logic học tiếp (Incremental Learning) nếu đã có model cũ
    checkpoint_path = os.path.join(model_save_dir, "phobert_genz_v1.pt")
    if os.path.exists(checkpoint_path):
        print(f">>> Tìm thấy model cũ tại {checkpoint_path}. Đang load để học tiếp...")
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE), strict=False)

    model = model.to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0, 
        num_training_steps=total_steps
    )
    loss_fn = nn.CrossEntropyLoss().to(DEVICE)

    # 4. Bắt đầu vòng lặp huấn luyện
    print(f"\n>>> BẮT ĐẦU FINE-TUNE PHOBERT TRÊN {len(dataset)} COMMENT...")
    print("-" * 30)
    
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        train_acc, train_loss = train_epoch(
            model, 
            train_loader, 
            loss_fn, 
            optimizer, 
            DEVICE, 
            scheduler, 
            len(dataset)
        )
        print(f"Loss: {train_loss:.4f} | Accuracy: {train_acc:.4f}")
        print("-" * 30)

    # 5. Lưu kết quả
    torch.save(model.state_dict(), checkpoint_path)
    print(f"\nHUẤN LUYỆN HOÀN TẤT!")
    print(f"Model đã được lưu tại: {checkpoint_path}")

if __name__ == "__main__":
    main()