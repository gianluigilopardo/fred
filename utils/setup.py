import importlib
import inspect


def setup_dataset(dataset_name):
    """Sets up the dataset.

    Args:
        dataset_name

    Returns:
        The training and testing sets.
    """

    # Construct the dataset path
    dataset_path = 'data.' + dataset_name

    # Import the dataset module
    mod = importlib.import_module(dataset_path)

    # Get the Dataset class from the module
    Dataset = getattr(mod, 'Dataset')

    # Create an instance of the Dataset class
    data = Dataset()

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = data.X_train, data.X_test, data.y_train, data.y_test

    class_names = data.class_names

    # Return the training and testing sets
    return X_train, X_test, y_train, y_test, class_names


def setup_model(model_name):
    """Sets up the model.

    Args:
        model_name.

    Returns:
        The model instance.
    """

    # Construct the model path
    model_path = 'models.' + model_name

    # Import the model module
    mod = importlib.import_module(model_path)

    # Get all the classes defined in the model module
    cls = inspect.getmembers(mod, inspect.isclass)

    # Filter the classes based on the given model argument
    filtered_cls = list(filter(lambda x: model_name.replace('_', '') == x[0].lower(), cls))

    # Extract the name of the desired model class
    model_name = filtered_cls[0][0]

    # Get the desired model class from the module
    Model = getattr(mod, model_name)

    # Create an instance of the model class
    model = Model()

    # Return the model instance
    return model


def setup_explainers(explainer_names, class_names, model, pos_dataset, args=None):
    """Sets up the explainers.

    Args:
        explainer_names

    Returns:
        The list of explainers instance.
    """
    explainers = {}

    for explainer_name in explainer_names:
        # Construct the model path
        explainer_path = 'explainers.' + explainer_name.lower() + '_explainer'

        # Import the model module
        exp = importlib.import_module(explainer_path)

        # Get the desired explainer class from the module
        Explainer = getattr(exp, explainer_name)

        # Create an instance of the model class
        if explainer_name == 'FRED_pos':
            explainer = Explainer(explainer_name, class_names, model, pos_dataset, args)
        else:
            explainer = Explainer(explainer_name, class_names, model, args)

        explainers[explainer_name] = explainer

    # Return the model instance
    return explainers
