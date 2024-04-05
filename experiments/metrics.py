import numpy as np
from sklearn.metrics import auc

from experiments.utils import jaccard_similarity

                                                       
def compute_sufficiency(example, tokens, confidence, explanation, explainer, exp, model, repetitions, k):
    """
    Calculates the sufficiency of the given explanation for the given example.

    Args:
        tokens: The tokens in the example.
        confidence: The confidence of the prediction for the example.
        explanation: The explanation.

    Returns:
        The sufficiency of the explanation for the example.
    """

    # Generate a list of word indices
    ids = list(range(len(tokens)))

    # Replace tokens at indices that are not in tokens_explainer with 'UNK'
    tokens_sufficiency = np.array(tokens)
    tokens_sufficiency[np.setdiff1d(ids, explanation)] = 'UNK'

    # Join the tokens back into an example string
    example_sufficiency = ' '.join(tokens_sufficiency)

    # Calculate the sufficiency score
    # RMK: we explain positive predictions
    sufficiency = confidence - model.predict_proba([example_sufficiency])[0, 1]

    return sufficiency


def compute_comprehensiveness(example, tokens, confidence, explanation, explainer, exp, model, repetitions, k):
    """
    Calculates the comprehensiveness of the given explanation for the given example.

    Args:
        tokens: The tokens in the example.
        confidence: The confidence of the prediction for the example.
        explanation: The ids in the explanation.

    Returns:
        The comprehensiveness of the explanation for the example.
    """

    # Generate a list of word indices
    ids = list(range(len(tokens)))

    # Replace tokens at indices specified by tokens_explainer with 'UNK'
    tokens_comprehensiveness = np.array(tokens)
    tokens_comprehensiveness[explanation] = 'UNK'

    # Join the tokens back into an example string
    example_comprehensiveness = ' '.join(tokens_comprehensiveness)

    # Calculate the comprehensiveness score
    # RMK: we explain positive predictions
    comprehensiveness = confidence - model.predict_proba([example_comprehensiveness])[0, 1]

    return comprehensiveness


def compute_robustness(example, tokens, confidence, explanation, explainer, exp, model, repetitions, k):
    """
    Calculates the robustness of the given explanation for the given example.

    Args:
        example: The text example to calculate the robustness for.
        explanation: The explanation for the given example.
        exp: The name of the explainer that was used to generate the explanation.

    Returns:
        The robustness of the given explanation for the given example.
    """

    # Initialize a list to store the similarity between the explanation and the adversarial example explanations.
    similarity = []

    # For the given number of repetitions, generate an adversarial example and calculate the similarity between
    # the explanation and the adversarial example explanation.
    for _ in range(repetitions):
        new_exp = explainer.explain(example)
        __, new_explanation = explainer.get_top_features(example, new_exp, k)
        similarity.append(jaccard_similarity(explanation, new_explanation))

    # Calculate the robustness as the mean of the similarities.
    robustness = np.mean(similarity)

    # Return the robustness.
    return robustness


def compute_aucmorf(example, tokens, confidence, explanation, explainer, exp, model, repetitions, k, max_len=20):
    """
    Calculates the AUC-MoRF of the given explanation for the given example.

    Args:
        exp: The explainer to be evaluated.
        example: The example to be explained.
        max_len (int, optional): The maximum number of tokens to consider for masking. Defaults to 20.

    Returns:
        The AUC-MoRF of the explanation for the example.
    """

    # Calculate the confidence of the model's prediction on the example.
    confidence = model.predict_proba([example])[0, 1]

    # Get the importance of each word in the explanation.
    _, ids_by_importance = explainer.get_ranked_tokens(example, exp)

    max_len = min(max_len, len(tokens))  # Limit max_len to the number of tokens

    if ids_by_importance is not None:
        curve = {0: 1}
        masked_scores = []

        rem_tokens = tokens.copy()

        # Iterate over the tokens in explanation, considering max_len
        for i, id in enumerate(ids_by_importance):
            if i >= max_len:
                break

            rem_tokens[id] = 'UNK'
            rem_doc = ' '.join(rem_tokens)

            # Predict confidence on masked text and store normalized score
            score = model.predict_proba([rem_doc])[0, 1] / confidence
            masked_scores.append(score)
            curve[i + 1] = score

        # Fill remaining curve with last score (if any)
        if masked_scores:
            last_score = masked_scores[-1]
            for i in range(len(masked_scores), max_len):
                curve[i + 1] = last_score
        else:
            # No importance scores, set AUC to 1
            auc_score = 1
            return auc_score

        # Convert masked scores to NumPy array and calculate AUC
        curve_array = np.array(list(curve.items()))

        auc_score = auc(curve_array[:, 0], curve_array[:, 1]) / max_len

    else:
        auc_score = 1  # No importance scores, set AUC to 1

    return auc_score
