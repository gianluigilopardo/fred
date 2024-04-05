import argparse

DATASETS = ['restaurants', 'yelp', 'tweets', 'imdb']
MODELS = ['logistic_classifier', 'tree_classifier', 'forest_classifier', 'roberta', 'distilbert']

METRICS = ['sufficiency', 'comprehensiveness', 'robustness', 'aucmorf']
EXPLAINERS = ['FRED', 'FRED_pos', 'LIME', 'SHAP', 'Anchor']


def parse_args():
    """Parses the command-line arguments.

  Returns:
    A namespace object containing the parsed arguments.
  """

    parser = argparse.ArgumentParser(description='Evaluate Explainers')

    # Required arguments
    parser.add_argument('--dataset',
                        type=str,
                        choices=DATASETS,
                        required=True,
                        help='Name of the dataset.')
    parser.add_argument('--model',
                        type=str,
                        choices=MODELS,
                        required=True,
                        help='Name of the model.')

    # Optional arguments
    parser.add_argument('--metrics',
                        type=str,
                        nargs='*',
                        default=METRICS,
                        help='The metrics to use for comparison.')
    parser.add_argument('--explainers',
                        type=str,
                        nargs='*',
                        default=EXPLAINERS,
                        help='The explainers to compare.')

    parser.add_argument('--perturb_proba',
                        type=float,
                        nargs='*',
                        default=0.5,
                        help='FRED sampling perturb_proba.')

    parser.add_argument('--sort_data',
                        type=bool,
                        default=True,
                        help='If true, sort dataset by smallest document.')
    parser.add_argument('--repetitions',
                        type=int,
                        default=10,
                        help='Number of iterations to compute robustness.')

    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help='Seed for reproducibility.')
    parser.add_argument('--start_doc',
                        type=int,
                        default=0,
                        help='First document of the corpus to evaluate.')
    parser.add_argument('--end_doc',
                        type=int,
                        default=100,
                        help='Last document of the corpus to evaluate.')
    parser.add_argument('--n_save',
                        type=int,
                        default=5,
                        help='Save results every n_save instances.')

    return parser.parse_args()
