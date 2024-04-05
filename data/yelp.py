import pandas as pd
import os
import re

from sklearn.model_selection import train_test_split

DATASET_PATH = 'data'


class Dataset:
    def __init__(self):
        if not os.path.exists(os.path.join(DATASET_PATH, 'positive_negative_reviews_yelp.csv')):
            raise IOError

        self.df = pd.read_csv(os.path.join(DATASET_PATH, 'positive_negative_reviews_yelp.csv'), sep='|')
        X = [self.preprocess(x) for x in list(self.df["text"])]
        y = list(self.df["stars"])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y)
        self.class_names = ['negative', 'positive']

    def preprocess(self, x):
        # light preprocessing: remove symbols, remove double spaces, lowercase
        x = re.sub('[^a-zA-Z\d\s]', '', x).lower()
        return re.sub(r"\s+", ' ', x)
