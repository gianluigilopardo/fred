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
