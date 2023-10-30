import pandas as pd
import os
# import kaggle
import re

DATASET_PATH = 'data'


class Dataset:
    def __init__(self):
        if not os.path.exists(os.path.join(DATASET_PATH, 'positive_negative_reviews_yelp.csv')):
            raise IOError

        self.df = pd.read_csv(os.path.join(DATASET_PATH, 'positive_negative_reviews_yelp.csv'), sep='|')
        self.X = [self.preprocess(x) for x in list(self.df["text"])]
        self.y = list(self.df["stars"])

    def preprocess(self, x):
        # light preprocessing: remove symbols, keep spaces, lowercase
        return re.sub('[^a-zA-Z\d\s]', '', x).lower()
