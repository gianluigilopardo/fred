from fred import explainer
import numpy as np

from explainers.base_explainer import Explainer
from explainers import utils


class FRED(Explainer):
    """
    FRED explainer for text classification.

    Inherits from the base Explainer class and provides methods for explaining
    text predictions using LIME.
    """

    def __init__(self, name, class_names, model, args=None):
        """
        Initializes the FRED explainer.

        Args:
            name (str): Name of the explainer.
            class_names (list): List of class names for the classifier.
            model (callable): The text classification function to be explained.
        """
        super().__init__(name, class_names, model)
        self.name = name
        self.model = model
        self.explainer = explainer.Fred(class_names=class_names, classifier_fn=model.predict_proba)
        if args:
            self.perturb_proba = args.perturb_proba
        else:
            self.perturb_proba = None

    def explain(self, example):
        """
        Explains a text instance using FRED.

        Args:
            example (str): The text instance to be explained.

        Returns:
            object: The explanation object returned by FRED.
        """
        if self.perturb_proba is not None:
            return self.explainer.explain_instance(str(example), perturb_proba=self.perturb_proba)
        return self.explainer.explain_instance(str(example))

    def get_ranked_tokens(self, example, fred_exp):
        """
        Retrieves tokens ranked by importance from a FRED explanation.

        Args:
            example (str): The text instance.
            fred_exp (object): The FRED explanation object.

        Returns:
            list, list:
                - List of tokens in the order of importance (highest to lowest).
                - List of corresponding feature indices in the same order.
        """
        tokens = np.array(example.split())
        ids_by_importance = [key for key, value in fred_exp.token_drops_ids.items() if value >= 0]
        return list(tokens[ids_by_importance]), ids_by_importance

    def get_top_features(self, example, fred_exp, k=10):
        """
        Extracts the top k most important tokens from a FRED explanation.

        Args:
            example (str): The text instance.
            fred_exp (object): The FRED explanation object.
            k (int, optional): The number of top features to return. Defaults to 10.

        Returns:
            list, list:
                - List of the top k most important tokens.
                - List of corresponding feature indices for the top k tokens.
        """
        return fred_exp.best, fred_exp.best_ids

    def get_tokens_importance(self, fred_exp):
        """
        Returns a dictionary mapping each token to its importance score.

        Ranks tokens based on their importance scores in the LIME explanation
        and returns a dictionary with the top k tokens (by importance).

        Args:
            fred_exp (object): The FRED explanation object.

        Returns:
            dict: Dictionary containing the top k tokens and their importance scores.
        """
        return fred_exp.token_drops
