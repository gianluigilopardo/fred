import random
import numpy as np

import spacy
from textblob import TextBlob

from collections import defaultdict

nlp = spacy.load("en_core_web_lg")  # Load spaCy model for tokenization and POS tagging

SEED = 42
np.random.seed(SEED)
random.seed(SEED)


class Replacer:

    def __init__(self, dataset, sentiment_mode='opposite'):
        """
        Initializes the Replacer class.

        Args:
            dataset (list): A list of strings representing documents used to build the category lists.
            sentiment_mode (str, optional): Controls how sentiment is used for replacement suggestions.
                                             Options: "opposite" (default), "same", or None (considers all tokens).
        """
        self.dataset = dataset
        self.sentiment_mode = sentiment_mode
        self.category_list = self.generate_category_lists(dataset)

    def generate_category_lists(self, documents):
        """
        Builds a dictionary mapping part-of-speech (POS) tags to sets of positive and negative tokens.

        Uses spaCy for tokenization and TextBlob for sentiment polarity.

        Args:
            documents (list): A list of strings representing documents.

        Returns:
            dict: A dictionary mapping POS tags to tuples of (positive_tokens, negative_tokens).
        """
        category_lists = defaultdict(lambda: (set(), set()))  # Default to empty sets for positive and negative tokens
        for doc in documents:
            doc = nlp(str(doc))
            for token in doc:
                category = token.pos_  # POS tag (e.g., noun, verb)
                sentiment = TextBlob(token.lemma_).sentiment.polarity  # Sentiment polarity (-1 to 1)
                if sentiment >= -0.1:
                    category_lists[category][0].add(token.text)  # Add to positive_tokens
                if sentiment <= +0.1:
                    category_lists[category][1].add(token.text)  # Add to negative_tokens
        return category_lists

    def generate_category_map(self, text):
        """
        Generates a dictionary mapping token positions (ids) to candidate replacement tokens.

        Considers sentiment for candidate replacements if sentiment_mode is set.

        Args:
            text (str): The input text string.

        Returns:
            dict: A dictionary mapping token ids in the text to lists of candidate replacement tokens.
        """
        category_lists = self.category_list
        tokens = text.split()
        text = ' '.join(tokens)  # Ignore double spaces
        b = len(tokens)
        doc = nlp(str(text))
        category_map = {}
        for id, token in enumerate(doc):
            category = token.pos_
            sentiment = TextBlob(token.text).sentiment.polarity
            # Access appropriate token list based on sentiment mode
            if self.sentiment_mode == "opposite":
                replacements = category_lists[category][1] if sentiment >= 0 else category_lists[category][0]
            elif self.sentiment_mode == 'same':
                replacements = category_lists[category][0] if sentiment >= 0 else category_lists[category][1]
            else:
                replacements = category_lists[category][0].union(category_lists[category][1])  # All tokens

            # do not replace token with itself
            if token.text in replacements:
                replacements.remove(token.text)

            if id > 0 and 1 <= len(replacements) <= 3:
                # if too less replacements, let us take a random set
                replacements.union(category_map[random.randint(0, id-1)])
            elif len(replacements) == 0:
                replacements.add('UNK')

            category_map[id] = replacements
        return category_map

    def replace_token(self, token_id, size, category_map):
        """
        Suggests replacement tokens for a specific token in the text.

        Args:
            token_id (int): The position (id) of the token to replace in the text.
            size (int): The number of replacement suggestions to return.

        Returns:
            list: A list of size elements, each containing a suggested replacement token.
        """

        replaced_token = random.choices([w for w in category_map[token_id]], k=size)
        return replaced_token
