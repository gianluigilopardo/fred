import numpy as np
from tabulate import tabulate
import pandas as pd
import os


# Mapper function to convert Lime explanation to a dictionary
def lime_dict(x):
    x = x.as_list()
    return {x[i][0]: x[i][1] for i in range(len(x))}


# Mapper function to extract positive words from Lime explanation
def lime_list(x):
    x = x.as_list()
    return [x[i][0] for i in range(len(x)) if x[i][1] >= 0]


# Mapper function to extract word IDs for positive words from Lime explanation
def lime_id_list(x):
    x = x.as_map()[1]
    return [x[i][0] for i in range(len(x)) if x[i][1] >= 0]


# Mapper function to extract word IDs for all words from Lime explanation
def lime_all_id_list(x):
    x = x.as_map()[1]
    return [x[i][0] for i in range(len(x))]


# Jaccard similarity between two lists
def jaccard_similarity(list1, list2):
    list1, list2 = list(list1), list(list2)
    if not list1 or not list2:
        intersection = 0
    else:
        intersection = len(list(set(list1).intersection(list2)))
    if not list1 and not list2:
        intersection = 1
        union = 1
    else:
        union = (len(set(list1)) + len(set(list2))) - intersection
    return float(intersection) / union


def show_results(results, results_path=None, average=False):
    """Prints the results in tabular form for any explainer, without the 'example' column.

  Args:
    results: A dictionary of results, where each key is an explainer and each value is a dictionary of metrics.
    results_path: The path to the final results CSV file.
    average: Whether to print the average and standard deviation of the metrics.
  """

    # Create a list of headers for the table.
    headers = ["Explainer"] + [metric for metric in list(results[list(results.keys())[0]].keys()) if
                               metric not in ['example', 'explanation']]

    # Create a list of rows for the table.
    rows = []
    for explainer in results:
        row = [explainer]
        for metric in results[explainer].keys():
            if metric not in ['example', 'explanation']:
                if len(results[explainer][metric]) > 0:
                    if average:
                        row.append(
                            str(f'{format(np.average(results[explainer][metric]), ".3f")} '
                                f'({format(np.std(results[explainer][metric]), ".2f")})'))
                    else:
                        row.append(f'{format(results[explainer][metric][-1], ".3f")}')
        rows.append(row)

    # Print the table.
    if average:
        print('-' * len(tabulate(rows, headers=headers)))
        print(tabulate(rows, headers=headers))
        print('-' * len(tabulate(rows, headers=headers)))
        print()
        # Save the results to a CSV file.
        df = pd.DataFrame(rows, columns=headers)
        df.to_csv(os.path.join(results_path, 'results.csv'), index=False)
    else:
        print()
        print(f'\t{tabulate(rows, headers=headers)}')
