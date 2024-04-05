import re

from datasets import load_dataset


dataset = load_dataset("imdb")
# https://huggingface.co/datasets/imdb


class Dataset:
    def __init__(self):
        self.X_train = [self.preprocess(x) for x in list(dataset['train']['text'])]
        self.y_train = list(dataset['train']['label'])

        self.X_test = [self.preprocess(x) for x in list(dataset['test']['text'])]
        self.y_test = list(dataset['test']['label'])

        self.class_names = ['negative', 'positive']

    def preprocess(self, x):
        # light preprocessing: remove symbols, remove double spaces, lowercase
        x = re.sub('[^a-zA-Z\d\s]', '', x).lower()
        return re.sub(r"\s+", ' ', x)
