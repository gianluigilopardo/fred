import pandas as pd
import os
# import kaggle
import re

DATASET_PATH = 'data'


class Dataset:
    def __init__(self):
        if not os.path.exists(os.path.join(os.getcwd(), DATASET_PATH, 'restaurants.tsv')):
            raise IOError

        self.df = pd.read_csv(os.path.join(DATASET_PATH, 'restaurants.tsv'), sep='\t')
        self.X = [self.preprocess(x) for x in list(self.df["Review"])]
        self.y = list(self.df["Liked"])

    def preprocess(self, x):
        # light preprocessing: remove symbols, keep spaces, lowercase
        return re.sub('[^a-zA-Z\d\s]', '', x).lower()
