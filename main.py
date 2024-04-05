import os
import numpy as np
import datetime
import json
import torch

from sklearn.metrics import accuracy_score, confusion_matrix

from experiments.evaluation import Evaluation
from utils.args import *
from utils.setup import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    """The main function."""

    # Parse the command-line arguments
    args = parse_args()

    # Set the random seed
    np.random.seed(args.seed)

    # Set up the dataset
    X_train, X_test, y_train, y_test, class_names = setup_dataset(args.dataset)

    # Set up the model
    model = setup_model(args.model)

    # Define run name/ID based on current time
    run = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")

    # Set up path to store results
    results_path = os.path.join('.', 'results', args.dataset, args.model, run)
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # Train the model
    model.train(X_train, y_train)

    # Evaluate model performance
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print(f'Confusion matrix: \n {cm}')
    accuracy = accuracy_score(y_test, y_pred)
    print(f'accuracy: {accuracy}')

    # Save experiment information
    info = {
        "args": {k: v for k, v in args.__dict__.items() if k != "_get_kwargs"},
        "confusion_matrix": cm.tolist(),
        "accuracy": accuracy,
    }
    json_data = json.dumps(info, indent=4)
    with open(os.path.join(results_path, 'info.json'), "w") as file:
        file.write(json_data)

    # Evaluate explainability on the positively predicted test examples
    corpus = np.asarray(X_test)[model.predict(X_test) == 1]
    corpus = corpus[np.char.count(corpus, ' ') > 2]

    if args.sort_data:
        corpus = sorted(corpus, key=lambda x: len(x.split()))

    corpus = corpus[args.start_doc:args.end_doc]

    # explainers
    np.random.shuffle(X_test)
    explainers = setup_explainers(args.explainers, class_names, model, X_test[:100], args)

    # Create an evaluation object
    evaluation = Evaluation(corpus=corpus, model=model, results_path=results_path,
                            explainers=explainers, args=args)

    # Evaluate the explainability metrics for the given explainers
    evaluation.evaluate()


if __name__ == '__main__':
    main()
