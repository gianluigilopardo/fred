import inspect
import os
import importlib
import numpy as np
import datetime
import json
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from experiments.evaluation import Evaluation
from utils.args import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def setup_dataset(args):
    """Sets up the dataset.

    Args:
        args: The parsed command-line arguments.

    Returns:
        The training and testing sets.
    """

    # Construct the dataset path
    dataset_path = 'data.' + args.dataset

    # Import the dataset module
    mod = importlib.import_module(dataset_path)

    # Get the Dataset class from the module
    Dataset = getattr(mod, 'Dataset')

    # Create an instance of the Dataset class
    data = Dataset()

    # Extract the dataframe, feature matrix, and target vector from the dataset
    df, X, y = data.df, data.X, data.y

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # Return the training and testing sets
    return X_train, X_test, y_train, y_test


def setup_model(args):
    """Sets up the model.

    Args:
        args: The parsed command-line arguments.

    Returns:
        The model instance.
    """

    # Construct the model path
    model_path = 'models.' + args.model

    # Import the model module
    mod = importlib.import_module(model_path)

    # Get all the classes defined in the model module
    cls = inspect.getmembers(mod, inspect.isclass)

    # Filter the classes based on the given model argument
    filtered_cls = list(filter(lambda x: args.model.replace('_', '') == x[0].lower(), cls))

    # Extract the name of the desired model class
    model_name = filtered_cls[0][0]

    # Get the desired model class from the module
    Model = getattr(mod, model_name)

    # Create an instance of the model class
    model = Model()

    # Return the model instance
    return model


def main():
    """The main function."""

    # Parse the command-line arguments
    args = parse_args()

    # Set the random seed
    np.random.seed(args.seed)

    # Set up the dataset
    X_train, X_test, y_train, y_test = setup_dataset(args)

    # Set up the model
    model = setup_model(args)

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

    # Evaluate explainability on the correctly predicted test examples
    corpus = np.asarray(X_test)[model.predict(X_test) == 1]
    if args.sort_data:
        corpus = sorted(corpus, key=lambda x: len(x.split()))

    # Create an evaluation object
    evaluation = Evaluation(corpus=corpus[args.start_doc:], model=model, results_path=results_path, args=args)

    # Evaluate the explainability metrics for the given explainers
    evaluation.evaluate()


if __name__ == '__main__':
    main()
