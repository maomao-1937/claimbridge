"""Local pretrained NLI inference; model weights are fetched on first use."""

from __future__ import annotations


MODEL_ID = "cross-encoder/nli-MiniLM2-L6-H768"
LABELS = ("contradiction", "entailment", "neutral")


class LocalNLI:
    def __init__(self) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
        self.model.eval()

    def score(self, premise: str, hypothesis: str) -> dict[str, float]:
        import torch

        inputs = self.tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512)
        with torch.inference_mode():
            probabilities = self.model(**inputs).logits.softmax(dim=-1)[0].tolist()
        return dict(zip(LABELS, probabilities, strict=True))
