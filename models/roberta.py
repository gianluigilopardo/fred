from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
import numpy as np

from .base_model import Model

# Load tokenizer and model, create trainer
model_name = "siebert/sentiment-roberta-large-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)


# Create class for data preparation
class SimpleDataset:
    def __init__(self, tokenized_texts):
        self.tokenized_texts = tokenized_texts

    def __len__(self):
        return len(self.tokenized_texts["input_ids"])

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.tokenized_texts.items()}


class Roberta(Model):
    def __init__(self):
        super().__init__()
        self.model = Trainer(model=model)

    def train(self, x, y, **kwargs):
        pass

    def predict(self, x):
        # Tokenize texts and create prediction data set
        tokenized_texts = tokenizer(x, truncation=True, padding=True)
        pred_dataset = SimpleDataset(tokenized_texts)
        # Run predictions
        predictions = self.model.predict(pred_dataset)
        preds = predictions.predictions.argmax(-1)
        return preds

    def predict_proba(self, x):
        # Tokenize texts and create prediction data set
        tokenized_texts = tokenizer(x, truncation=True, padding=True)
        pred_dataset = SimpleDataset(tokenized_texts)
        # Run predictions
        predictions = self.model.predict(pred_dataset)
        scores = (np.exp(predictions[0]) / np.exp(predictions[0]).sum(-1, keepdims=True))
        return scores
