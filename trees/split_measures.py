import numpy as np


def evaluate_measures(sample):
    """Calculate measure of split quality (each node separately).

    Please use natural logarithm (e.g. np.log) to evaluate value of entropy measure.

    Parameters
    ----------
    sample : a list of integers. The size of the sample equals to the number of objects in the current node. The integer
    values are equal to the class labels of the objects in the node.

    Returns
    -------
    measures - a dictionary which contains three values of the split quality.
    Example of output:

    {
        'gini': 0.1,
        'entropy': 1.0,
        'error': 0.6
    }

    """
    d = {}
    for k in sample:
        if k in d:
            d[k] += 1
        else:
            d[k] = 1
    node = np.array(list(d.keys()))
    obj_num = np.array(list(d.values()))
    py = obj_num / sum(obj_num)
    gini = (py * (1 - py)).sum()
    entropy = (py * (- np.log(py))).sum()
    error = (py * (1 - py.max())).sum()
    measures = {'gini': gini, 'entropy': entropy, 'error': error}
    return measures
