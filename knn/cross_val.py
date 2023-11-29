import numpy as np
import typing
from collections import defaultdict


def kfold_split(num_objects: int,
                num_folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    res = []
    all_indices = np.arange(0, num_objects)
    fold_length = int(num_objects / num_folds)
    for i in range(0, fold_length * (num_folds - 1), fold_length):
        fold = np.arange(i, i + fold_length)
        res.append((np.delete(np.copy(all_indices), fold), fold))

    res.append((np.arange(fold_length * (num_folds - 1)), (np.arange(fold_length * (num_folds - 1), num_objects))))
    return res


def knn_cv_score(X: np.ndarray, y: np.ndarray, parameters: dict[str, list],
                 score_function: callable,
                 folds: list[tuple[np.ndarray, np.ndarray]],
                 knn_class: object) -> dict[str, float]:
    res = dict()
    for normalizer in parameters['normalizers']:
        scaler = normalizer[0]
        for i in parameters['n_neighbors']:
            for metric in parameters['metrics']:
                for weight in parameters['weights']:

                    clf = knn_class(n_neighbors=i, weights=weight, metric=metric)
                    score = []
                    for f in folds:
                        train, test = f
                        train_normed = X[train]
                        test_normed = X[test]
                        if scaler:
                            scaler.fit(X[train])
                            train_normed = scaler.transform(X[train])
                            test_normed = scaler.transform(X[test])

                        clf.fit(X=train_normed, y=y[train])
                        y_predict = clf.predict(test_normed)
                        score.append(score_function(y[test], y_predict))

                    res[(normalizer[1], i, metric, weight)] = np.mean(score)
    return res
