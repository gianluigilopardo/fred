import re

from datasets import load_dataset

from sklearn.model_selection import train_test_split

dataset = load_dataset("tweets_hate_speech_detection")


# https://huggingface.co/datasets/tweets_hate_speech_detection


class Dataset:
    def __init__(self):
        X = [self.preprocess(x) for x in list(dataset['train']['tweet'])]
        y = list(dataset['train']['label'])

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y)

        self.class_names = ['no-hate-speech', 'hate-speech']

    def preprocess(self, x):
        # light preprocessing: remove symbols, remove double spaces, lowercase
        x = re.sub('[^a-zA-Z\d\s]', '', x).lower()
        return re.sub(r"\s+", ' ', x)
