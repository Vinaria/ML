import numpy as np
from typing import Tuple


def sum_non_neg_diag(x):
    y = np.diag(x)[np.diag(x) >= 0]
    return np.sum(y) if y.size > 0 else -1


def are_multisets_equal(x, y):
    x_vals, x_counts = np.unique(x, return_counts = True)
    y_vals, y_counts = np.unique(y, return_counts = True)
    if (np.shape(x_vals) != np.shape(y_vals)): return False 
    if (np.any(x_vals != y_vals)): return False 
    if (np.any(x_counts != y_counts)): return False
    return True


def max_prod_mod_3(x):
    y = np.array(x)
    y = (y[1:] * y[:-1])
    y = y[y % 3 == 0]
    return y.max() if y.size > 0 else -1


def convert_image(image, weights):
    image = np.array(image)
    weights = np.array(weights)
    image = np.rot90(image, axes = (2, 0))
    image = np.rot90(image, axes = (1, 2))
    return np.einsum('ijk, i -> ijk', image, weights).sum(axis = 0)


def rle_scalar(x, y):
    x = np.array(x)
    y = np.array(y)
    lenx = x[:, 1].sum()
    leny = y[:, 1].sum()
    
    if lenx != leny:
        return -1
    
    new_x = np.repeat(x[:, 0], x[:, 1])
    new_y = np.repeat(y[:, 0], y[:, 1])
    
    return np.dot(new_x, new_y)


def cosine_distance(X, Y):
    X = np.array(X)
    Y = np.array(Y)
    Z = np.dot(X, Y.T)
    norm_X = np.linalg.norm(X, axis = 1)
    norm_Y = np.linalg.norm(Y, axis = 1)
    Z[norm_X == 0] = 1
    Z[:, norm_Y == 0] = 1
    norm_X[norm_X == 0] = 1
    norm_Y[norm_Y == 0] = 1
    
    
    return Z / (np.outer(norm_X, norm_Y))
    
