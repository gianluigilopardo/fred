import gensim.downloader as api
import numpy as np


class Replacer:
    """
    A class for representing and using word embeddings.

    This class allows you to load pre-trained word embedding models and use them to replace words with similar words.

    Example usage:

    ```python
    wv = Replacer()
    replaced_word = wv.replace_word("dog")

    print(replaced_word)

    """

    def __init__(self, dataset='glove-wiki-gigaword-100'):
        """
        Initializes the Replacer class.

        Args:
            dataset (str, optional): The name of the pre-trained word embedding model to load. Defaults to 'glove-wiki-gigaword-100'.
        """

        self.wv = self.get_model(dataset)

    def get_model(self, dataset):
        """
        Loads a pre-trained word embedding model.

        Args:
            dataset (str): The name of the pre-trained word embedding model to load.

        Returns:
            gensim.models.Word2Vec: A Gensim Word2Vec model.
        """

        wv = api.load(dataset)
        return wv

    def most_dissimilar(self, word, topn=10):
        """
        Returns the most dissimilar words to a given word, according to the cosine similarity metric.

        Args:
            word (str): The word to find the most dissimilar words for.
            topn (int, optional): The number of most dissimilar words to return. Defaults to 10.

        Returns:
            tuple[list[str], list[float]]: A tuple containing a list of the most dissimilar words and a list of the corresponding cosine similarity values.
        """

        wv = self.wv

        # Get the word vector for the given word.
        word_vector = wv[word]

        # Get all of the word vectors in the vocabulary.
        vectors_all = wv.vectors

        # Calculate the cosine similarity between the given word vector and all of the other word vectors in the
        # vocabulary.
        cosine_similarity_values = wv.cosine_similarities(word_vector, vectors_all)

        # Get the indices of the least similar words.
        least_similar_word_indices = cosine_similarity_values.argsort()[0:topn]

        # Get the most dissimilar words.
        most_dissimilar_words = [wv.index_to_key[i] for i in least_similar_word_indices]

        # Return the most dissimilar words and their corresponding cosine similarity values.
        return most_dissimilar_words, cosine_similarity_values[0:topn]

    def replace_word(self, word, size):
        """
        Replaces a word with a word with a probability proportional to the cosine similarity.

        Args:
            word (str): The word to be replaced.

        Returns:
            str: A replaced word.
        """

        wv = self.wv
        vocabulary = set(wv.key_to_index.keys())

        # Check if the word is in the vocabulary.
        if word not in vocabulary:
            return ["UNK"] * size

        # Get the most dissimilar words to the original word.
        most_dissimilar_words, word_similarity = self.most_dissimilar(word, topn=10)
        probas = np.exp(-word_similarity) / np.sum(np.exp(-word_similarity), axis=0)
        # print(most_dissimilar_words)
        # print(probas)

        # Select a word from the most dissimilar words with a probability inversely proportional to the cosine
        # similarity.
        replaced_word = np.random.choice(most_dissimilar_words, p=probas, size=size)

        return replaced_word




