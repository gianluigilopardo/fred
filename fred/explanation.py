import numpy as np

from IPython.core.display import display, HTML


class Explanation:
    """
    Represents an explanation for a model's prediction on a specific example.

    This class holds information about the input example, the features that contribute
    most to the prediction, and optionally counterfactual examples that can help understand
    the model's reasoning.
    """

    def __init__(self, example: str, best: list, drop_best: float, best_ids: list,
                 token_drops_ids: dict, label: int, sample: np.ndarray,
                 token_removals: list, sample_mask: np.ndarray,
                 pos: bool, classifier_fn: callable,
                 eps: float, confidence: float,
                 class_names: list = None,
                 verbose: bool = True):
        """
        Initializes an Explanation object.

        Args:
            example (str): The original input text.
            best (list): List of tokens contributing most to the model's confidence.
            drop_best (float): The drop in model confidence when removing the 'best' tokens.
            best_ids (list): Indices of the 'best' tokens in the original text.
            token_drops_ids (dict): Dictionary mapping tokens to their drop in confidence when removed.
            label (int): The predicted label for the example.
            sample (list or np.ndarray): The internal representation of the example used by the model.
            token_removals (list): List of tokens removed for individual feature importance analysis.
            sample_mask (np.ndarray): Mask indicating which features were perturbed.
            pos (bool): Flag indicating if explanation is based on POS tagging.
            classifier_fn (callable): The function used to make predictions with the model.
            class_names (list, optional): List of human-readable names for the labels (default: None).
            verbose (bool, optional): Flag for printing explanations during initialization (default: True).
        """

        self.example = example
        self.pos = pos  # Flag for POS-based explanation
        self.tokens = np.array(example.split())  # Tokenized input text
        self.best = best
        self.drop = drop_best
        self.best_ids = best_ids
        self.label = label
        self.sample = np.array(sample)

        self.eps = eps
        self.confidence = confidence

        self.token_removals = token_removals  # Optional: removed tokens for feature importance
        self.sample_mask = sample_mask  # Optional: mask for perturbed features

        self.classifier_fn = classifier_fn
        self.class_names = class_names
        self.verbose = verbose

        self.probas = self.classifier_fn([example])
        self.pred = self.probas.argmax(1)[0]

        self.mode = 'pos' if self.pos else 'mask'

        unranked_token_drops = list(zip(self.tokens, token_drops_ids.values()))
        self.unranked_token_drops = unranked_token_drops

        # Sort token_drops by drop in confidence (descending order)
        self.token_drops_ids = dict(sorted(token_drops_ids.items(), key=lambda item: item[1], reverse=True))
        self.ranked_ids = list(self.token_drops_ids.keys())
        self.ranked_tokens = self.tokens[self.ranked_ids]

        # self.token_drops = dict(sorted(unranked_token_drops.items(), key=lambda item: item[1], reverse=True))
        self.token_drops = list(zip(self.ranked_tokens, self.token_drops_ids.values()))

        # Loop through the dictionary
        saliency = []
        max_len = 10
        for (token, weight) in self.token_drops:
            if len(saliency) >= max_len:
                break
            # Format the value with three decimal places using f-string
            saliency.append((token, np.round(weight, 3)))

        if verbose:
            print()
            print(f'FRED mode: \'{self.mode} sampling\'.')
            print(f'Example to explain: \n\t\'{self.example}\'')
            print(f'Original prediction: \'{class_names[self.pred] if class_names else self.pred}\'')
            print(f'Average confidence over the sample: {round(confidence, 3)}')
            print()
            if self.class_names:
                print(f'Explaining class \'{self.class_names[label]}\':')
            else:
                print(f'Explaining class \'{self.label}\':')
            print(f'The minimal subset of tokens that make the confidence drop by {eps*100}% if perturbed is \n\t{self.best}')
            print()
            print(f'Saliency weights: \n\t{saliency}')

    def counterfactual(self, counter_label=None, verbose=False, k=1):
        """
        Finds counterfactual examples with minimal perturbation.

        This method identifies samples within the internal representation space that
        lead to a different prediction (or a specific predicted label) with minimal
        changes to the original input.

        Args:
            counter_label (int, optional): The label for which to find counterfactuals (default: None - find different predictions).
                If provided, finds counterfactuals predicted as 'counter_label'.
            verbose (bool, optional): Flag for printing detailed information about counterfactuals (default: False).
            k (int, optional): The number of counterfactual examples to return (default: 1).

        Returns:
            tuple: A tuple containing the counterfactual examples (perturbed samples) and the corresponding perturbed tokens.
        """

        # Get original prediction
        pred = self.pred

        # Sort samples by total mask value (perturbation)
        sorted_sample_ids = np.argsort(self.sample_mask.sum(1))

        # Predict on sorted samples (increasing perturbation order)
        preds = self.classifier_fn(self.sample[sorted_sample_ids]).argmax(1)

        # Identify counterfactual sample indices based on prediction comparison
        if counter_label is None:
            # Find samples with different predictions compared to original
            diff_preds = np.where(pred != preds)[0]
        else:
            # Find samples predicted as the specified counter_label
            diff_preds = np.where(preds == self.class_names.index(counter_label))[0]

        # Select top k counterfactual samples
        counter_sample = self.sample[sorted_sample_ids[diff_preds]][:k]

        # Identify perturbed tokens in the top counterfactual sample
        perturbed_tokens = [list(np.array(self.tokens)[self.sample_mask[sorted_sample_ids[diff_preds]][i] == 1])
                            for i in range(k)]

        if verbose:
            print(f'\nCounterfactual explanation for the example\n\t\'{self.example}\'')
            print()
            print(f'FRED mode: \'{self.mode} sampling\'.')
            print(f'Original prediction: \'{self.class_names[self.pred] if self.class_names else self.pred}\'')
            print()
            if counter_label:
                print(f'Sample(s) with minimal perturbation predicted as \'{counter_label}\':')
            else:
                print(f'Sample(s) with minimal perturbation predicted differently:')
            print()
            print(f'{counter_sample}')
            print()
            print(f'Perturbed tokens: \n\t{perturbed_tokens}')

        return counter_sample, perturbed_tokens

    def html_explanation(self):
        """
        Visualizes the explanation with tokens colored based on importance.

        Args:
            self (Explanation): An instance of the Explanation class.

        Returns:
            HTML: The HTML content for the visualization.
        """

        html_content = """
        <style>
            .token { display: inline-block; padding: 2px 5px; margin: 2px; }
        </style>
            """

        # Color mapping function (adjust the range and formula as needed)
        def get_color(drop_value, scale):
            """
            Maps drop value to a color between red (negative) and green (positive) with white in the middle,
            with an optional alpha parameter for transparency.

            Args:
                drop_value (float): The importance score for a token.
                scale (float): Scaling factor for drop_value.

            Returns:
                str: The RGB color code in the format "rgba(r, g, b, a)".
            """

            color_value = (255, 100, 255)

            alpha = abs(drop_value) / scale

            # Handle cases outside the threshold
            if drop_value <= 0:
                return f"rgba({color_value[0]}, 0, 0, {alpha})"  # Red with varying intensity and alpha
            elif drop_value >= 0:
                return f"rgba(0, {color_value[1]}, 0, {alpha})"  # Green with varying intensity and alpha

        # Build the HTML with color-coded tokens
        scale = np.max(np.abs(list(self.token_drops_ids.values())))
        for (token, weight) in self.unranked_token_drops:
            color = get_color(weight, scale)
            html_content += f"<span class='token' style='background-color: {color};'>{token}</span>"
        return HTML(html_content)

