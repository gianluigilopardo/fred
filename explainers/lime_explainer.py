from lime import lime_text
import numpy as np

from explainers.base_explainer import Explainer
from explainers import utils


class LIME(Explainer):
    """
    LIME explainer for text classification.

    Inherits from the base Explainer class and provides methods for explaining
    text predictions using LIME.
    """

    def __init__(self, name, class_names, model, args=None):
        """
        Initializes the LIME explainer.

        Args:
            name (str): Name of the explainer.
            class_names (list): List of class names for the classifier.
            model (callable): The text classification function to be explained.
        """
        super().__init__(name, class_names, model)
        self.name = name
        self.model = model
        self.explainer = lime_text.LimeTextExplainer(class_names=class_names, bow=False)

    def explain(self, example):
        """
        Explains a text instance using LIME.

        Args:
            example (str): The text instance to be explained.

        Returns:
            object: The explanation object returned by LIME.
        """
        return self.explainer.explain_instance(str(example), classifier_fn=self.model.predict_proba)

    def get_ranked_tokens(self, example, lime_exp):
        """
        Retrieves tokens ranked by importance from a LIME explanation.

        Args:
            example (str): The text instance.
            lime_exp (object): The LIME explanation object.

        Returns:
            list, list:
                - List of tokens in the order of importance (highest to lowest).
                - List of corresponding feature indices in the same order.
        """
        tokens = np.array(example.split())
        ids_by_importance = utils.lime_id_list(lime_exp)
        return list(tokens[ids_by_importance]), ids_by_importance

    def get_top_features(self, example, lime_exp, k=10):
        """
        Extracts the top k most important tokens from a LIME explanation.

        Args:
            example (str): The text instance.
            lime_exp (object): The LIME explanation object.
            k (int, optional): The number of top features to return. Defaults to 10.

        Returns:
            list, list:
                - List of the top k most important tokens.
                - List of corresponding feature indices for the top k tokens.
        """
        tokens_by_importance, ids_by_importance = self.get_ranked_tokens(example, lime_exp)
        return tokens_by_importance[:k], ids_by_importance[:k]

    def get_tokens_importance(self, lime_exp):
        """
        Returns a dictionary mapping each word to its importance score.

        Ranks tokens based on their importance scores in the LIME explanation
        and returns a dictionary with the top k tokens (by importance).

        Args:
            lime_exp (object): The LIME explanation object.
            k (int, optional): The number of top tokens to return. Defaults to 10.

        Returns:
            dict: Dictionary containing the top k tokens and their importance scores.
        """
        weights = utils.lime_dict(lime_exp)
        return dict(list(sorted(weights.items(), key=lambda item: item[1], reverse=True)))
