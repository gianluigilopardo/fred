class Explainer:
    """
    Base class for text classification explainers.

    Provides a foundation for building modular explainers with common methods
    for analyzing text data.
    """

    def __init__(self, name, class_names, classifier_fn, args=None):
        """
        Initializes the explainer.

        Args:
            name: The name of the explainer.
            class_names: A list of class names.
            classifier_fn (object): The text classification classifier_fn to be explained.
        """
        self.name = name
        self.class_names = class_names
        self.classifier_fn = classifier_fn

    def explain(self, example):
        """
        Returns a dictionary containing the explanation.

        The specific explanation format will depend on the explainer implementation.

        Raises:
            NotImplementedError: If not implemented in the subclass.
        """
        raise NotImplementedError("explain() method not implemented in subclass")

    def get_ranked_words(self):
        """
        Returns a list of words ranked by importance.

        The ranking criteria (e.g., SHAP values, LIME weights) might vary based on the explainer.

        Raises:
            NotImplementedError: If not implemented in the subclass.
        """
        raise NotImplementedError("get_ranked_words() method not implemented in subclass")

    def get_top_features(self, k=10):
        """
        Returns the top k most important words (subset of ranked words).

        Args:
            k (int, optional): The number of top features to return. Defaults to 10.

        Raises:
            NotImplementedError: If not implemented in the subclass.
        """
        ranked_words = self.get_ranked_words()
        return ranked_words[:k]

