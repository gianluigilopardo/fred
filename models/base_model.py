import pickle


class Model:
    def __init__(self):
        self.model = None

    def train(self, x, y, **kwargs):
        pass

    def load_model(self, filename):
        self.model = pickle.load(open(filename, 'rb'))

    def save_model(self, filename):
        pickle.dump(self.model, open(filename, 'wb'))
