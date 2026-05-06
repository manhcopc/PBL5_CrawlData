from threading import Lock
from typing import List, Optional

from app.core.config import settings


class PhobertService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PhobertService, cls).__new__(cls)
            cls._instance._boot()
        return cls._instance

    def _boot(self) -> None:
        self.tokenizer = None
        self.model = None
        self.is_ready = False
        self.load_error: Optional[str] = None
        self.device = "unknown"
        self._load_lock = Lock()

    def ensure_ready(self) -> bool:
        if self.is_ready:
            return True

        with self._load_lock:
            if self.is_ready:
                return True

            print("[PhoBERT Service] Lazy-loading PhoBERT sentiment model...")
            try:
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.tokenizer = AutoTokenizer.from_pretrained(settings.PHOBERT_PATH)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    settings.PHOBERT_PATH
                ).to(self.device)
                self.model.eval()
                self.is_ready = True
                self.load_error = None
                print("[PhoBERT Service] PhoBERT ready.")
                return True
            except Exception as exc:
                self.is_ready = False
                self.load_error = str(exc)
                print(f"[PhoBERT Service] Load failed: {exc}")
                return False

    def analyze_sentiment(self, reviews: List[str]) -> float:
        if not reviews:
            return 0.0
        if not self.ensure_ready() or self.model is None or self.tokenizer is None:
            raise RuntimeError(f"PhoBERT Service is offline: {self.load_error}")

        import torch
        from pyvi import ViTokenizer

        processed_texts = [ViTokenizer.tokenize(text) for text in reviews]
        inputs = self.tokenizer(
            processed_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            positive_scores = probs[:, 1].cpu().tolist()

        return sum(positive_scores) / len(positive_scores)


nlp_service = PhobertService()

