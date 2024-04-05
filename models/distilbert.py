from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer
import numpy as np

from .base_model import Model

# Load tokenizer and model, create trainer
model_name = "distilbert-base-uncased"  # Specify DistilBERT model here
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name)


# Create class for data preparation
class SimpleDataset:
    def __init__(self, tokenized_texts):
        self.tokenized_texts = tokenized_texts

    def __len__(self):
        return len(self.tokenized_texts["input_ids"])

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.tokenized_texts.items()}


class DistilBert(Model):
    def __init__(self):
        super().__init__()
        self.model = Trainer(model=model)

    def train(self, x, y, **kwargs):
        pass  # You can implement training logic here

    def predict(self, x):
        probas = self.predict_proba(x)
        preds = probas.argmax(-1)
        return preds

    def predict_proba(self, x):
        x = [str(t) for t in x]
        # Tokenize texts and create prediction data set
        tokenized_texts = tokenizer(x, truncation=True, padding=True)
        pred_dataset = SimpleDataset(tokenized_texts)
        # Run predictions
        predictions = self.model.predict(pred_dataset)
        scores = (np.exp(predictions[0]) / np.exp(predictions[0]).sum(-1, keepdims=True))
        return scores
