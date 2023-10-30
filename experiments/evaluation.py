import time
import os
import numpy as np
import pandas as pd
import spacy
from sklearn.metrics import auc

# explainers
from anchor import anchor_text
from lime import lime_text
from fred import explainer

from experiments import utils

pd.set_option('display.max_columns', None)

EXPLAINERS = ['FRED', 'FRED_star', 'FRED_pos', 'FRED_pos_star', 'FRED_test', 'FRED_pos_test', 'LIME', 'Anchors']


def init_explainer(name, class_names, model):
    """Initializes an explainer based on its name.

  Args:
    name: The name of the explainer.
    class_names: A list of class names.
    model: A machine learning model.

  Returns:
    An explainer object.
  """

    if name == "Anchors":
        nlp = spacy.load('en_core_web_lg')
        return anchor_text.AnchorText(nlp, class_names)
    elif name == "LIME":
        return lime_text.LimeTextExplainer(class_names=class_names, bow=False)
    elif name in ["FRED", "FRED_star", "FRED_test"]:
        return explainer.Fred(class_names=class_names, classifier_fn=model.predict_proba)
    elif name in ["FRED_pos", "FRED_pos_star", "FRED_pos_test"]:
        return explainer.Fred(class_names=class_names, classifier_fn=model.predict_proba, pos=1)
    else:
        raise ValueError(f"Unknown explainer name: {name}")


class Evaluation:
    """
    This class is used to evaluate different explainers on a given corpus.

    Args:
        corpus: A list of text examples.
        model: A machine learning model.
        results_path: The path where the results should be saved.
    """

    def __init__(self, corpus, model, results_path, args):
        self.corpus = corpus
        self.model = model
        self.results_path = results_path
        self.metrics = args.metrics
        self.repetitions = args.repetitions
        self.eps = args.eps
        self.n_save = args.n_save
        self.beam_size = args.beam_size
        self.explainers = EXPLAINERS if 'full' in args.explainers else args.explainers
        self.nlp = spacy.load('en_core_web_lg')  # Anchors needs it

        self.explainer = {}
        # Initialize the explainers
        class_names = ["NEGATIVE", "POSITIVE"]  # binary classification
        for explainer_name in self.explainers:
            self.explainer[explainer_name] = init_explainer(explainer_name, class_names, model)

    def evaluate(self):
        """
        Evaluates the different explainers on the corpus and saves the results to the results_path.
        """

        # Initialize the results
        results = {}
        for exp in self.explainers:
            results[exp] = {}
            for metric in self.metrics:
                results[exp][metric] = []
            results[exp]['example'] = []
            results[exp]['explanation'] = []
            results[exp]['time'] = []
            results[exp]['proportion'] = []

        # Evaluate each explainer
        for i, example in enumerate(self.corpus):
            confidence = self.model.predict_proba([example])[0, 1]
            print(f'({i + 1} / {len(self.corpus)}) Example: {example}\n\tconfidence: {confidence}')
            for exp in self.explainers:
                explanation, explanation_time = self.get_explanation(exp, example)
                words = np.array([x.text for x in self.nlp(str(example))],
                                 dtype='|U80') if exp == 'Anchors' else example.split()
                results[exp]['example'].append(example)
                results[exp]['time'].append(explanation_time)
                results[exp]['proportion'].append(len(explanation) / len(words))
                results[exp]['explanation'].append(np.array(words)[explanation])
                print(f'\t{exp}: {np.array(words)[explanation]}')

                # Calculate the metric
                for metric in self.metrics:
                    metric_value = self.calculate_metric_for_example(metric, example, words, confidence, explanation,
                                                                     exp)
                    results[exp][metric].append(metric_value)

                # store everything each n_save instances
                if (i + 1) % self.n_save == 0:
                    df = pd.DataFrame(results[exp])
                    df.to_csv(os.path.join(self.results_path, f'{exp}.csv'))

            utils.show_results(results)
            print()

            # store everything each n_save instances
            if (i + 1) % self.n_save == 0:
                print(f'Partial results for first {i + 1} instances out of {len(self.corpus)}:')
                utils.show_results(results, self.results_path, average=True)

        # Save the results
        for exp in self.explainers:
            df = pd.DataFrame(results[exp])
            df.to_csv(os.path.join(self.results_path, f'{exp}.csv'))

        utils.show_results(results, self.results_path, average=True)

    def get_explanation(self, exp, example):
        """
        Gets an explanation for the given example using the given explainer.

        Args:
            exp: The name of the explainer to use.
            example: The text example to get an explanation for.

        Returns:
            A tuple of two elements:
                * explanation: A list of the ids of the words in the example that are important for the prediction.
                * explanation_time: The time it took to generate the explanation, in seconds.
        """

        start_time = time.time()
        explanation = []

        # Get the explanation from the explainer.
        if exp in ['FRED', 'FRED_pos', 'FRED_test', 'FRED_pos_test']:
            explanation = self.explainer[exp].explain_instance(example, beam_size=self.beam_size).best_ids
        elif exp in ['FRED_star', 'FRED_pos_star']:
            explanation = self.explainer[exp].explain_instance(example, eps=0.2, beam_size=self.beam_size).best_ids
        elif exp == 'LIME':
            words = example.split()
            lime_exp = self.explainer[exp].explain_instance(str(example), self.model.predict_proba)
            num_features = int(np.ceil(len(words) * 0.05))  # top 5%
            explanation = utils.lime_id_list(lime_exp)[:num_features]
        elif exp == 'Anchors':
            explanation = self.explainer[exp].explain_instance(str(example), self.model.predict).features()
        else:
            raise ValueError(f'Explainer {exp} not implemented.')

        explanation_time = time.time() - start_time

        return explanation, explanation_time

    def get_words_importance(self, exp, example):
        # Get the words ranked by importance from the explainer.
        # RMK: we only consider positive here
        if exp in ['FRED', 'FRED_pos']:
            explanation = self.explainer[exp].explain_instance(example, beam_size=self.beam_size)
            ids_by_importance = [key for key, value in explanation.word_drops_ids.items() if value >= 0]
        elif exp == 'LIME':
            words = example.split()
            lime_exp = self.explainer[exp].explain_instance(str(example), self.model.predict_proba,
                                                            num_features=len(words))
            ids_by_importance = utils.lime_id_list(lime_exp)
        else:
            # for other explain does not make sense
            ids_by_importance = None
        return ids_by_importance

    def calculate_metric_for_example(self, metric, example, words, confidence, explanation, exp):
        """
        Calculates the given metric for the given example using the given explainer.

        Args:
            metric: The name of the metric to calculate.
            example: The text example to calculate the metric for.
            explanation: The explanation for the given example.
            exp: The name of the explainer that was used to generate the explanation.

        Returns:
            The value of the metric for the given example.
        """
        proportion = len(explanation) / len(words)

        if metric == 'comprehensiveness':
            metric_value = self.compute_comprehensiveness(words, confidence, explanation)
        elif metric == 'sufficiency':
            metric_value = self.compute_sufficiency(words, confidence, explanation)
        elif metric == 'robustness':
            metric_value = self.compute_robustness(example, explanation, exp)
        elif metric == 'aucmorf':
            metric_value = self.compute_aucmorf(exp, example)
        else:
            raise ValueError(f'Metric {metric} not implemented.')
        return metric_value

    def compute_sufficiency(self, words, confidence, explanation):
        """
        Calculates the sufficiency of the given explanation for the given example.

        Args:
            words: The words in the example.
            confidence: The confidence of the prediction for the example.
            explanation: The explanation.

        Returns:
            The sufficiency of the explanation for the example.
        """

        # Generate a list of word indices
        ids = list(range(len(words)))

        # Replace words at indices that are not in words_explainer with 'UNK'
        words_sufficiency = np.array(words)
        words_sufficiency[np.setdiff1d(ids, explanation)] = 'UNK'

        # Join the words back into an example string
        example_sufficiency = ' '.join(words_sufficiency)

        # Calculate the sufficiency score
        # RMK: we explain positive predictions
        sufficiency = confidence - self.model.predict_proba([example_sufficiency])[0, 1]

        return sufficiency

    def compute_comprehensiveness(self, words, confidence, explanation):
        """
        Calculates the comprehensiveness of the given explanation for the given example.

        Args:
            words: The words in the example.
            confidence: The confidence of the prediction for the example.
            explanation: The ids in the explanation.

        Returns:
            The comprehensiveness of the explanation for the example.
        """

        # Generate a list of word indices
        ids = list(range(len(words)))

        # Replace words at indices specified by words_explainer with 'UNK'
        words_comprehensiveness = np.array(words)
        words_comprehensiveness[explanation] = 'UNK'

        # Join the words back into an example string
        example_comprehensiveness = ' '.join(words_comprehensiveness)

        # Calculate the comprehensiveness score
        # RMK: we explain positive predictions
        comprehensiveness = confidence - self.model.predict_proba([example_comprehensiveness])[0, 1]

        return comprehensiveness

    def compute_robustness(self, example, explanation, exp):
        """
        Calculates the robustness of the given explanation for the given example.

        Args:
            example: The text example to calculate the robustness for.
            explanation: The explanation for the given example.
            exp: The name of the explainer that was used to generate the explanation.

        Returns:
            The robustness of the given explanation for the given example.
        """
        repetitions = self.repetitions

        # Initialize a list to store the similarity between the explanation and the adversarial example explanations.
        similarity = []

        # For the given number of repetitions, generate an adversarial example and calculate the similarity between
        # the explanation and the adversarial example explanation.
        for k in range(repetitions):
            new_explanation, _ = self.get_explanation(exp, example)
            similarity.append(utils.jaccard_similarity(explanation, new_explanation))

        # Calculate the robustness as the mean of the similarities.
        robustness = np.mean(similarity)

        # Return the robustness.
        return robustness

    def compute_aucmorf(self, exp, example):
        """
        Calculates the AUC-MoRF of the given explanation for the given example.

        Args:
            exp: The explainer to be evaluated.
            example: The example to be explained.

        Returns:
            The AUC-MoRF of the explanation for the example.
        """

        # Calculate the confidence of the model's prediction on the example.
        confidence = self.model.predict_proba([example])[0, 1]

        # Get the importance of each word in the explanation.
        ids_by_importance = self.get_words_importance(exp, example)

        words = example.split()
        if ids_by_importance:
            # Create a copy of the example text.
            rem_words = words.copy()

            # Initialize the AUC-MoRF curve.
            curve = {0: 1}

            # Iterate over the words in the explanation, in order of importance.
            for i, id in enumerate(ids_by_importance):
                # Mask out the current word in the example text.
                rem_words[id] = 'UNK'
                rem_doc = ' '.join(rem_words)

                # Calculate the confidence of the model's prediction on the masked example text.
                score = self.model.predict_proba([rem_doc])[0, 1]

                # Normalize the score by the confidence of the model's prediction on the original example text.
                score = score / confidence

                # Add the current score to the AUC-MoRF curve.
                curve[i + 1] = score

            m = len(curve)
            for i in range(m - 1, len(words)):
                curve[i + 1] = curve[m - 1]

            # Convert the AUC-MoRF curve to a NumPy array.
            curve_array = np.array(list(curve.items()))

            # Compute the scaled area under the curve
            auc_score = auc(curve_array[:, 0], curve_array[:, 1]) / len(words)
        else:
            auc_score = -1
        return auc_score
