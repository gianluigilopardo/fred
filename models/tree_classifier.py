from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from .base_model import Model


class TreeClassifier(Model):
    def __init__(self):
        super().__init__()
        self.model = None
        self.vectorizer = TfidfVectorizer()

    def train(self, x_train, y_train, **kwargs):
        # Transform the training data into TF-IDF feature vectors
        train_vectors = self.vectorizer.fit_transform(x_train)

        # Train the decision tree model
        self.model = DecisionTreeClassifier()
        self.model.fit(train_vectors, y_train)
        # TODO: save accuracy

    def predict(self, x):
        # Transform input using the trained vectorizer
        x_vectors = self.vectorizer.transform(x)

        # Make predictions using the trained model
        return self.model.predict(x_vectors)

    def predict_proba(self, x):
        # Transform input using the trained vectorizer
        x_vectors = self.vectorizer.transform(x)

        # Compute predicted probabilities using the trained model
        return self.model.predict_proba(x_vectors)

    #
    # def load_model(self, filename):
    #     self.model = pickle.load(open(filename, 'rb'))
    #
    # def save_model(self, filename):
    #     pickle.dump(self.model, open(filename, 'wb'))
