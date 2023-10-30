class Explanation:

    def __init__(self, words, best, drop_best, best_ids, word_drops_ids, label, class_names=None, verbose=True):
        """
        Initializes an Explanation object.

        Args:
            words (list): The list of words in the input text.
            best (list): The subset of words with the highest drop in model confidence.
            drop_best (float): The drop in model confidence corresponding to the best subset of words.
            best_ids (list): The indices of the best subset of words in the input text.
            word_drops_ids (dict): A dictionary mapping individual words to their corresponding drop in confidence.
                           The keys are the words, and the values are the corresponding drop in confidence.
            label (int): the label to explain
            class_names (list): A list of label names.
            verbose (bool): If print
        """

        self.words = words
        self.best = best
        self.drop = drop_best
        self.best_ids = best_ids
        self.label = label

        word_drops = dict(zip(words, word_drops_ids.values()))

        # Sort the word_drops dictionary by drop in confidence, in descending order
        self.word_drops_ids = dict(sorted(word_drops_ids.items(), key=lambda item: item[1], reverse=True))
        self.word_drops = dict(sorted(word_drops.items(), key=lambda item: item[1], reverse=True))

        if verbose:
            if class_names:
                print(f'Explaining class \'{class_names[label]}\':')
            else:
                print(f'Explaining class \'{label}\':')
            print(best)
