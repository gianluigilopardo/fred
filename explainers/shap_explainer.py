from shap import Explainer as SHAPExplainer
import numpy as np
from shap.maskers import Text as TextMasker

from sklearn.preprocessing import LabelEncoder

from explainers.base_explainer import Explainer


class SHAP(Explainer):
    """
    SHAP explainer for text classification.

    Inherits from the base Explainer class and provides methods for explaining
    text predictions using SHAP (SHapley Additive exPlanations).
    """

    def __init__(self, name, class_names, model, args=None):
        """
        Initializes the SHAP explainer.

        Args:
            name (str): Name of the explainer.
            class_names (list): List of class names for the classifier.
            model (callable): The text classification function to be explained.
        """
        super().__init__(name, class_names, model)

        self.name = name
        self.model = model
        # Assuming your model takes text as input and returns a probability distribution
        masker = TextMasker()
        self.explainer = SHAPExplainer(self.model.predict_proba, masker=masker)

    def tokenizer(self, text):
        """
        Function to tokenize text for SHAP explainer.

        You can implement your desired tokenization logic here (e.g., using spaCy).

        Args:
            text (str): The text to be tokenized.

        Returns:
            list: List of tokens representing the text.
        """
        # Replace with your preferred tokenization method (e.g., using spaCy)
        return text.split()  # Simple word splitting for demonstration

    def explain(self, example):
        """
        Explains a text instance using SHAP.

        Args:
            example (str): The text instance to be explained.

        Returns:
            object: The explanation object returned by SHAP.
        """
        shap_values = self.explainer([example])

        return shap_values  # Explanation object may vary depending on SHAP version

    def get_ranked_tokens(self, example, shap_values):
        """
        Retrieves tokens ranked by importance (SHAP value) from a SHAP explanation.

        Args:
            example (str): The text instance.
            shap_values (object): The SHAP explanation object.

        Returns:
            list, list:
                - List of tokens in the order of importance (highest SHAP value to lowest).
                - List of corresponding feature indices in the same order.
        """
        tokens = example.split()

        feature_importance = shap_values.values[0][:, 1]  # Assuming shap_values is a tuple

        # Assuming feature_importance is a numpy array
        # Sort tokens and importance together by importance (descending)
        ranked_indices = list(np.argsort(feature_importance)[::-1])
        ranked_tokens = [tokens[i] for i in ranked_indices]
        # sorted_importance = feature_importance[sorted_indices]

        return ranked_tokens, ranked_indices

    def get_top_features(self, example, shap_values, k=10):
        """
        Extracts the top k most important tokens from a SHAP explanation.

        Args:
            example (str): The text instance.
            shap_values (object): The SHAP explanation object.
            k (int, optional): The number of top features to return. Defaults to 10.

        Returns:
            list, list:
                - List of the top k most important tokens.
                - List of corresponding feature indices for the top k tokens.
        """
        ranked_tokens, ranked_indices = self.get_ranked_tokens(example, shap_values)
        return ranked_tokens[:k], ranked_indices[:k]

    def get_tokens_importance(self, shap_exp):
        """
        Returns a dictionary mapping each word to its importance score.

        Ranks tokens based on their importance scores in the SHAP explanation
        and returns a dictionary with the top k tokens (by importance).

        Args:
            shap_exp (object): The SHAP explanation object.
            k (int, optional): The number of top tokens to return. Defaults to 10.

        Returns:
            dict: Dictionary containing the top k tokens and their importance scores.
        """
        feature_importance = shap_exp.values[0][:, 1]  # Assuming shap_values is a tuple

        # Assuming feature_importance is a numpy array
        # Sort tokens and importance together by importance (descending)
        tokens = shap_exp.data[0]
        tokens_score = {tokens[i]: feature_importance[i] for i in range(len(tokens))}
        return dict(list(sorted(tokens_score.items(), key=lambda item: item[1], reverse=True)))
