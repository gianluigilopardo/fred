# Import necessary libraries
from anchor import anchor_text
import numpy as np  # For array operations
import spacy  # For natural language processing

from explainers.base_explainer import Explainer  # Import the base explainer class


# Define the Anchor explainer class, inheriting from Explainer
class Anchor(Explainer):
    """
    Anchor explainer for text classification.

    Inherits from the base Explainer class and provides methods for explaining
    text predictions using the Anchor algorithm.
    """

    # Initialize the Anchor explainer
    def __init__(self, name, class_names, model, args=None):
        """
        Initializes the Anchor explainer.

        Args:
            name (str): Name of the explainer.
            class_names (list): List of class names for the classifier.
            model (callable): The text classification function to be explained.
        """

        # Call the parent class constructor
        super().__init__(name, class_names, model)

        self.name = name
        self.model = model
        # Load the spaCy model
        self.nlp = spacy.load('en_core_web_lg')

        # Create the AnchorText explainer instance
        self.explainer = anchor_text.AnchorText(class_names=class_names, nlp=self.nlp)

    # Explain a text instance using Anchor
    def explain(self, example):
        """
        Explains a text instance using Anchor.

        Args:
            example (str): The text instance to be explained.

        Returns:
            object: The explanation object returned by Anchor.
        """

        # Call the explain_instance method of the Anchor explainer
        return self.explainer.explain_instance(str(example), self.model.predict)

    # Retrieve tokens ranked by importance from an Anchor explanation
    def get_ranked_tokens(self, example, anchor_exp):
        """
        Retrieves tokens ranked by importance from an Anchor explanation.

        Args:
            example (str): The text instance.
            anchor_exp (object): The Anchor explanation object.

        Returns:
            list, list:
                - List of tokens in the order of importance.
                - List of corresponding feature indices in the same order.
        """

        # Split the text into tokens and convert to a NumPy array
        tokens = np.array([x.text for x in self.nlp(str(example))], dtype='|U80')
        # Get the feature indices representing important tokens from the explanation
        ids_by_importance = anchor_exp.features()

        # Extract the tokens corresponding to the important feature indices
        tokens_by_importance = tokens[ids_by_importance]

        return list(tokens_by_importance), ids_by_importance

    # Extract the top k most important tokens from an Anchor explanation
    def get_top_features(self, example, anchor_exp, k=10):
        """
        Extracts the top k most important tokens from an Anchor explanation.

        Args:
            example (str): The text instance.
            anchor_exp (object): The Anchor explanation object.
            k (int, optional): The number of top features to return. Defaults to 10.

        Returns:
            list, list:
                - List of the top k most important tokens.
                - List of corresponding feature indices for the top k tokens.
        """

        # Get the ranked tokens and feature indices
        tokens_by_importance, ids_by_importance = self.get_ranked_tokens(example, anchor_exp)

        # Return the top k tokens and their corresponding feature indices
        return tokens_by_importance[:k], ids_by_importance[:k]
