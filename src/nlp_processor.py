import torch.nn as nn
from transformers import AutoModel

class PhoBertSentimentClassifier(nn.Module):
    def __init__(self, n_classes=2):
        super(PhoBertSentimentClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained("vinai/phobert-base")
        
        self.drop = nn.Dropout(p=0.3)
        
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        # Đưa dữ liệu qua PhoBERT
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        pooled_output = outputs.pooler_output
        
        output = self.drop(pooled_output)
        return self.out(output)