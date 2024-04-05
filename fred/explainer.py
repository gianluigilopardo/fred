"""
FRED explainer

This class implements the FRED explainer algorithm for text classification.

FRED works by perturbing the input text and observing the change in the predicted output.
The tokens that have the biggest impact on the prediction are considered to be the most important
tokens for the explanation.
"""
import random

import numpy as np
import itertools
import re

from .explanation import Explanation
from fred.replacer import Replacer

SEED = 42
np.random.seed(SEED)
random.seed(SEED)


class Fred:

    def __init__(self, classifier_fn, class_names=None, mask='UNK', pos=False, pos_dataset=None):
        """
        Initializes a Fred explainer object.

        Args:
            class_names (list): A list of label names.
            classifier_fn (function): A function that takes in a text input and returns a probability distribution over classes.
            mask (str, optional): A string used to replace tokens during sampling. Defaults to 'UNK'.
            pos (bool, optional): A boolean flag indicating whether to use token2Vec replacement during sampling. Defaults to False.
            pos_dataset (list, optional): A list of documents used for token2Vec replacement (if pos=True).
        """
        self.class_names = class_names
        self.classifier_fn = classifier_fn
        self.mask = mask
        self.pos = pos
        if self.pos:
            self.replacer = Replacer(pos_dataset)  # Initialize replacer for POS replacement

    def generate_sample(self, example, max_len, perturb_proba, n_sample, alpha=0.99):
        """
        Generates a new sample by perturbing a subset of tokens in the example.

        Args:
            example (str): The input text to be perturbed.
            max_len (int): The maximum length of the perturbed subsets of tokens.
            perturb_proba (float, optional): The probability of perturbing each token. Defaults to 0.1.
            alpha (float, optional): The confidence level for determining the number of samples to generate.
                                     Defaults to 0.99.

        Returns:
            list: A list of new samples generated by perturbing a subset of tokens in the example.
        """
        tokens = example.split()  # Split the example into a list of tokens

        # Commented out the calculation of n_sample as it's not directly used and alpha is set to a fixed value
        # n_sample = int(np.ceil(np.log(1 - alpha) / np.log(1 - perturb_proba ** 10)))  # Number of samples to generate
        # n_sample = 5000  # Fixed number of samples for fair evaluation

        perturbed = np.empty((n_sample, len(tokens)), dtype='U80')  # Array to store perturbed samples
        perturbed[:] = tokens  # Initialize perturbed samples with original tokens

        x = np.random.uniform(0, 1, size=(n_sample, len(tokens)))  # Random values for each token

        # Masked indices where random value <= perturb_proba
        masked = np.where(x <= perturb_proba)

        sample_mask = np.zeros_like(x)
        sample_mask[masked] = 1
        sample_mask = sample_mask.astype(int)

        # Apply perturbations
        if self.pos:
            category_map = self.replacer.generate_category_map(example)
            for id in range(x.shape[1]):
                pert = masked[0][masked[1] == id]
                perturbed[pert, id] = self.replacer.replace_token(id, len(pert), category_map)
        else:
            perturbed[masked] = self.mask  # Replace with mask token

        sample = [' '.join(x) for x in perturbed]  # Convert back to list of strings

        token_removals = {j: [] for j in range(len(tokens))}  # Removed token indices per token position
        for i, j in enumerate(masked[1]):
            token_removals[j].append(masked[0][i])

        return sample, token_removals, sample_mask

    def generate_candidates(self, example, size):
        """
        Generates all possible combinations of tokens with a given size from the example.

        Args:
            example (str): The input text, example to be explained.
            size (int): The number of tokens in each combination.

        Returns:
            tuple: A tuple containing a list of candidate combinations and a list of their corresponding indices.
        """

        # Split the example into a list of tokens
        tokens = example.split()

        # Generate indices combinations using itertools.combinations
        comb_indices = itertools.combinations(range(len(tokens)), size)

        # Convert combinations to lists of indices
        candidate_idx = [[i for i in comb] for comb in comb_indices]

        # Generate token combinations based on the list of indices
        candidates = [list(comb) for comb in itertools.combinations(tokens, size)]

        return candidates, candidate_idx

    def beam_candidates(self, candidates_ids, token_drops, tokens, beam_size, max_missing_per_candidate=10):
        """
        Generates the next beam of candidates using a beam search algorithm.

        Args:
            candidates_ids (list): A list of indices for the candidate combinations of tokens.
            token_drops (dict): A dictionary containing token drops for each token index.
            tokens (list): A list of all tokens in the example.
            beam_size (int): The beam size.
            max_missing_per_candidate (int, optional): The maximum number of missing elements allowed per candidate. Defaults to 10.

        Returns:
            tuple: A tuple containing a list of candidate combinations and a list of their corresponding indices.
        """

        tokens = np.array(tokens)

        # Sort token_ids by token drops (higher drop is more important)
        token_ids = list(sorted(token_drops.items(), key=lambda item: item[1], reverse=True))
        token_ids = [w[0] for w in token_ids]  # Extract token indices

        # Select the top 'beam_size' candidates
        prev_candidates_ids = np.array(candidates_ids)[:beam_size]

        # Initialize the new list of candidates and their corresponding indices
        candidates = []
        new_candidates_ids = []

        # Iterate over the previous candidates
        for candidate in prev_candidates_ids:

            # Find missing elements not present in the candidate
            missing_elements = token_ids.copy()  # Use existing sorted list
            [missing_elements.remove(ic) for ic in candidate]
            missing_elements = np.array(missing_elements)

            # Limit missing elements to max_missing_per_candidate
            missing_elements = missing_elements[:max_missing_per_candidate]

            # If there are missing elements
            if len(missing_elements) > 0:
                # Replicate the candidate list for each missing element
                new_lists = np.tile(candidate, (len(missing_elements), 1))

                # Append the missing element to each replicated list
                new_lists = np.hstack((new_lists, missing_elements[:, np.newaxis]))

                # Extend the new_candidates_ids list with the new candidates
                new_candidates_ids.extend(new_lists)

                # Convert the new_candidates_ids to candidate tokens
                candidates.extend([tokens[i].tolist() for i in new_lists])

        # Return the new list of candidates and their corresponding indices
        return candidates, new_candidates_ids

    def compute_drop(self, candidates_ids, sample_drops, token_removals):
        """
        Computes the average drop in confidence for a candidate combination of tokens.

        Args:
            candidates_ids (list): A list of token indices representing the candidate combination.
            sample_drops (list): A list containing confidence drops for each perturbed example (indexed by token removal).
            token_removals (dict): A dictionary mapping token indices to lists of removed tokens in the corresponding perturbed examples.

        Returns:
            float: The average drop in confidence for the candidate combination.

        Raises:
            ValueError: If there are no common removed tokens for the candidate combination.
        """

        # Find the intersection of removed tokens across all tokens in the candidate
        removed_tokens = set.intersection(*[set(token_removals[i]) for i in candidates_ids])

        # Retrieve confidence drops for the common removed tokens
        drops = [sample_drops[i] for i in removed_tokens]

        # Compute the average drop in confidence
        return np.mean(drops)

    def explain_instance(self, example, perturb_proba=0.5, eps=0.15, max_len=10, beam_size=4,
                         n_sample = 5000,
                         label=None,
                         verbose=False):
        """
        Generates an explanation for a given input text.

        Args:
            example (str): The input text to be explained.
            eps (float, optional): The threshold for the drop in confidence to determine the best subset of tokens.
                                       Defaults to 0.15.
            max_len (int, optional): The maximum length of the perturbed subsets of tokens. Defaults to 10.
            label (int, optional): The label to explain. If None, the predicted label.
            beam_size (int, optional): The number of top candidate explanations to consider at each step (beam search). Defaults to 4.
            verbose (bool, optional): If True, print debugging information. Defaults to False.

        Returns:
            Explanation: An Explanation object that contains the best subset of tokens and its drop in confidence.
            :param perturb_proba:
        """
        
        example = re.sub('[^a-zA-Z\d\s]', '', example).lower()
        example = re.sub(r"\s+", ' ', example)
        
        # Split the example text into a list of tokens
        tokens = np.array(example.split())

        # Get the predicted label if not provided (assuming a classification task)
        if label is None:
            label = np.argmax(self.classifier_fn([example]))

        # Pre-compute confidence for efficiency (avoid redundant calls to the classifier)
        confidence = self.classifier_fn([example])[:, label]

        # Generate perturbed samples (modified tokens) and compute confidence drops efficiently
        sample, token_removals, sample_mask = self.generate_sample(example, perturb_proba=perturb_proba,
                                                                  n_sample=n_sample,  max_len=max_len)
        preds = self.classifier_fn(sample)[:, label]
        sample_drops = confidence - preds  # Vectorized computation for faster drop calculation

        # Initialize variables to track the best explanation
        best = []
        best_drop = 0
        best_ids = []
        best_drops_sum = 0
        token_drops = {}  # Dictionary to store token-level drops (efficiency)

        # Initialize a dictionary to store the drop in confidence for each candidate combination of tokens
        candidate_drops = {}

        # Initialize candidate explanations and their corresponding token indices
        candidates = [[w] for w in tokens]  # List of token lists (candidates)
        candidates_ids = [[k] for k in range(len(tokens))]  # List of token index lists

        # Iterate over candidate lengths (up to max_len)
        for size in range(1, max_len + 1):

            # Prune and sort candidates based on drop (early stopping)
            if candidate_drops:
                candidate_drops = dict(
                    sorted(candidate_drops.items(), key=lambda item: item[1], reverse=True)[:beam_size])
                sorted_ids = list(candidate_drops.keys())
                sorted_candidates_ids = np.array(candidates_ids)[sorted_ids]

                # Update candidates and their corresponding token indices based on the best drops
                candidates, candidates_ids = self.beam_candidates(sorted_candidates_ids, token_drops, tokens, beam_size)

            # Reset candidate drops for the current size
            candidate_drops = {}

            # Analyze each candidate explanation of the current size
            for i, candidate in enumerate(candidates):
                # Compute the drop in confidence for the current candidate explanation
                candidate_drops[i] = self.compute_drop(candidates_ids[i], sample_drops, token_removals)

                # Track token-level drops for single-token candidates
                if len(candidate) == 1:
                    token_drops[i] = candidate_drops[i]

                # Update the best explanation based on drop and token-level drops (combined metric)
                if candidate_drops[i] > best_drop:
                    best = candidate
                    best_drop = candidate_drops[i]
                    best_ids = candidates_ids[i]
                    best_drops_sum = np.sum([token_drops[k] for k in candidates_ids[i]])
                elif candidate_drops[i] == best_drop:
                    drops_sum = np.sum([token_drops[k] for k in candidates_ids[i]])
                    if drops_sum >= best_drops_sum:
                        best = candidate
                        best_drop = candidate_drops[i]
                        best_ids = candidates_ids[i]

            # If the drop in confidence for the best subset of tokens is greater than or equal to the threshold
            if best_drop >= eps * confidence:
                # Return an Explanation object that contains the best subset of tokens and its drop in confidence
                explanation = Explanation(example, best, best_drop, best_ids, token_drops,
                                          label, sample, token_removals, sample_mask,
                                          self.pos, self.classifier_fn,
                                          eps, confidence[0],
                                          self.class_names,
                                          verbose)
                return explanation

        # If the drop in confidence for the best subset of tokens is less than the threshold
        # Return an Explanation object that contains the best subset of tokens and its drop in confidence
        explanation = Explanation(example, best, best_drop, best_ids, token_drops,
                                  label, sample, token_removals, sample_mask,
                                  self.pos, self.classifier_fn,
                                  eps, confidence[0],
                                  self.class_names,
                                  verbose)
        return explanation
