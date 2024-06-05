import numpy as np

import sklearn
import sklearn.metrics
from sklearn.metrics import pairwise_distances


def silhouette_score(x, labels):
    '''
    :param np.ndarray x: Непустой двумерный массив векторов-признаков
    :param np.ndarray labels: Непустой одномерный массив меток объектов
    :return float: Коэффициент силуэта для выборки x с метками labels
    '''
    unique, counts = np.unique(labels, return_counts=True)

    if counts.shape == (1,):
        return 0

    dist1 = pairwise_distances(x)
    mask = np.array([labels] * len(labels)) == labels.reshape((-1, 1))
    dist1[~mask] = 0

    C = ((np.array([labels] * unique.size) == unique.reshape((-1, 1))) * counts.reshape((-1, 1))).sum(axis=0)

    s = np.zeros(x.shape[0])
    s[C > 1] = dist1[C > 1].sum(axis=1) / (C[C > 1] - 1)

    mask = np.array([labels] * unique.size) == unique.reshape((-1, 1))
    new_dist = np.repeat(pairwise_distances(x), unique.size, axis=0).reshape((x.shape[0], unique.size, x.shape[0]))
    new_mask = np.array([mask] * x.shape[0])
    new_dist[~new_mask] = 0

    class_distances = new_dist.sum(axis=2) / np.array([counts] * x.shape[0])

    indices = np.unique(labels, return_inverse=True)[1] + np.array(range(0, class_distances.size, class_distances.shape[1]))
    dist2 = np.delete(class_distances, indices).reshape((class_distances.shape[0], class_distances.shape[1] - 1))

    d = np.zeros(x.shape[0])
    d[C > 1] = (dist2[C > 1]).min(axis=1)

    sil = np.zeros(x.shape[0])
    mask = (C == 1) | (np.maximum(d, s) == 0)
    sil[~mask] = ((d - s)[~mask] / np.maximum(d, s)[~mask])

    sil_score = sil.sum() / x.shape[0]
    return sil_score


def bcubed_score(true_labels, predicted_labels):
    '''
    :param np.ndarray true_labels: Непустой одномерный массив меток объектов
    :param np.ndarray predicted_labels: Непустой одномерный массив меток объектов
    :return float: B-Cubed для объектов с истинными метками true_labels и предсказанными метками predicted_labels
    '''

    c_matrix = (np.array([predicted_labels] * predicted_labels.size) ==
                np.repeat(predicted_labels, predicted_labels.size).reshape((predicted_labels.size, predicted_labels.size))) * 1
    l_matrix = (np.array([true_labels] * true_labels.size) ==
                np.repeat(true_labels, true_labels.size).reshape((true_labels.size, true_labels.size))) * 1
    correctness = c_matrix * l_matrix

    precision = (correctness.sum(axis=1) / c_matrix.sum(axis=1)).mean()
    recall = (correctness.sum(axis=1) / l_matrix.sum(axis=1)).mean()
    score = 2 * (precision * recall) / (precision + recall)

    return score
