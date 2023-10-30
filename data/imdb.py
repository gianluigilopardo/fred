import pandas as pd
import os
# import kaggle
import re

DATASET_PATH = '.'  # penso sia meglio far scaricare manualmente il file piuttosto che fare tutto lo sbatti di kaggle


class Dataset:
    def __init__(self):
        # if not os.path.exists(os.path.join('.', 'Restaurant_Reviews.tsv')):
        #     kaggle.api.authenticate()
        #     kaggle.api.dataset_download_files('Restaurant-reviews',
        #                                       path='.', unzip=True)
        if not os.path.exists(os.path.join(DATASET_PATH, 'imdb.csv')):
            raise IOError

        self.df = pd.read_csv(os.path.join(DATASET_PATH, 'imdb.csv'), sep=',', nrows=10000)
        self.X = [self.preprocess(x) for x in list(self.df["review"])]
        self.y = self.df.sentiment.copy()
        self.y = self.y.replace('positive', 1)
        self.y = self.y.replace('negative', 0)
        self.y = list(self.y)

    def preprocess(self, x):
            # light preprocessing: remove symbols, keep spaces, lowercase
            return re.sub('[^a-zA-Z\d\s]', '', x).lower()
