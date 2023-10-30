import argparse

DATASETS = ['restaurants', 'yelp']
MODELS = ['logistic_classifier', 'tree_classifier', 'forest_classifier', 'roberta']

METRICS = ['sufficiency', 'comprehensiveness', 'robustness', 'aucmorf']
EXPLAINERS = ['FRED', 'FRED_star', 'FRED_pos', 'FRED_pos_star', 'LIME', 'Anchors']


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
    parser.add_argument('--sort_data',
                        type=bool,
                        default=True,
                        help='If true, sort dataset by smallest document.')
    parser.add_argument('--repetitions',
                        type=int,
                        default=5,
                        help='Number of iterations to compute robustness.')
    parser.add_argument('--beam_size',
                        type=int,
                        default=4,
                        help='The beam size to use for all FRED variants.')
    parser.add_argument('--eps',
                        type=float,
                        default=0.3,
                        help='The threshold for the drop in confidence for FRED_test and FRED_pos_test.')
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help='Seed for reproducibility.')
    parser.add_argument('--start_doc',
                        type=int,
                        default=0,
                        help='First document of the corpus to evaluate.')
    parser.add_argument('--n_save',
                        type=int,
                        default=5,
                        help='Save results every n_save instances.')

    return parser.parse_args()
