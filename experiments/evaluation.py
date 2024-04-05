import time
import os
import numpy as np
import pandas as pd
import spacy
from sklearn.metrics import auc
import random

import explainers
from fred import explainer
from experiments import utils
from experiments import metrics

map_metrics = {'sufficiency': metrics.compute_sufficiency,
               'comprehensiveness': metrics.compute_comprehensiveness,
               'robustness': metrics.compute_robustness,
               'aucmorf': metrics.compute_aucmorf}

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

pd.set_option('display.max_columns', None)


class Evaluation:

    def __init__(self, corpus, model, results_path, explainers, args):
        self.corpus = corpus
        self.model = model
        self.results_path = results_path

        self.metrics = args.metrics
        self.repetitions = args.repetitions
        self.n_save = args.n_save

        self.explainers = explainers
        self.explainer_names = list(explainers.keys())

    def evaluate(self):
        """
            Evaluates the different explainers on the corpus and saves the results to the results_path.
            """

        # Initialize the results
        results = {}
        for exp in self.explainer_names:
            results[exp] = {}
            for metric in self.metrics:
                results[exp][metric] = []
            results[exp]['example'] = []
            results[exp]['explanation'] = []
            results[exp]['time'] = []
            results[exp]['proportion'] = []

        # Evaluate each explainer
        for i, example in enumerate(self.corpus):

            words = example.split()
            example = ' '.join(words)  # avoid double spaces

            confidence = self.model.predict_proba([example])[0, 1]

            print(f'({i + 1} / {len(self.corpus)}) '
                  f'Example: {example}\n'
                  f'\tconfidence: {confidence}')

            for explainer in self.explainers.values():

                explainer_name = explainer.name
                print(f'\t{explainer_name}:')

                t0 = time.time()
                exp = explainer.explain(example)
                exp_time = time.time() - t0

                tokens = np.array([x.text for x in explainer.nlp(str(example))],
                                  dtype='|U80') if explainer_name == 'Anchor' else example.split()

                n_features = 10
                if explainer_name not in ['FRED', 'FRED_pos']:
                    n_features = len(results['FRED_pos']['explanation'][-1])

                top_tokens, explanation = explainer.get_top_features(example, exp, k=n_features)

                results[explainer_name]['example'].append(example)
                results[explainer_name]['time'].append(exp_time)
                results[explainer_name]['proportion'].append(len(explanation) / len(tokens))
                results[explainer_name]['explanation'].append(top_tokens)

                print(f'\t\t{top_tokens}')

                # Calculate the metric
                for metric in self.metrics:
                    metric_value = map_metrics[metric](example, tokens, confidence,
                                                       explanation, explainer, exp,
                                                       self.model, self.repetitions,
                                                       k=n_features)
                    results[explainer_name][metric].append(metric_value)

                # store everything each n_save instances
                if (i + 1) % self.n_save == 0:
                    df = pd.DataFrame(results[explainer_name])
                    df.to_csv(os.path.join(self.results_path, f'{explainer_name}.csv'))

            utils.show_results(results)
            print()

            # store everything each n_save instances
            if (i + 1) % self.n_save == 0:
                print(f'Partial results for first {i + 1} instances out of {len(self.corpus)}:')
                utils.show_results(results, self.results_path, average=True)

        # Save the results
        for explainer_name in self.explainer_names:
            df = pd.DataFrame(results[explainer_name])
            df.to_csv(os.path.join(self.results_path, f'{explainer_name}.csv'))

        utils.show_results(results, self.results_path, average=True)
