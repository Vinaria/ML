import numpy as np


class Preprocessor:

    def __init__(self):
        pass

    def fit(self, X, Y=None):
        pass

    def transform(self, X):
        pass

    def fit_transform(self, X, Y=None):
        pass


class MyOneHotEncoder(Preprocessor):

    def __init__(self, dtype=np.float64):
        super(Preprocessor).__init__()
        self.dtype = dtype

    def fit(self, X, Y=None):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: unused
        """
        self.data = []
        self.lens = [0]
        X = np.array(X)
        if len(X.shape) > 1:
            for feature in X.T:
                res = {}
                vals = np.unique(np.array(feature))
                for i in range(vals.size):
                    res[vals[i]] = i
                self.data.append(res)
                self.lens.append(len(vals))
        else:
            feature = X
            res = {}
            vals = np.unique(np.array(feature))
            for i in range(vals.size):
                res[vals[i]] = i
            self.data.append(res)
            self.lens.append(len(vals))

        print(self.data, self.lens)

    def transform(self, X):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        returns: transformed objects, numpy-array, shape [n_objects, |f1| + |f2| + ...]
        """
        res = np.zeros((X.shape[0], sum(self.lens)))
        X = np.array(X)
        if len(X.shape) > 1:
            for (index, feature) in enumerate(X.T):
                feature = np.array(feature)
                for i in range(feature.size):
                    res[i][sum(self.lens[:index + 1]) + self.data[index][feature[i]]] = 1
        else:
            feature = np.array(X)
            index = 0
            for i in range(feature.size):
                res[i][sum(self.lens[:index + 1]) + self.data[index][feature[i]]] = 1

        return res

    def fit_transform(self, X, Y=None):
        self.fit(X)
        return self.transform(X)

    def get_params(self, deep=True):
        return {"dtype": self.dtype}


class SimpleCounterEncoder:

    def __init__(self, dtype=np.float64):
        self.dtype = dtype

    def fit(self, X, Y):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: target for training objects, pandas-series, shape [n_objects,]
        """
        self.data = []
        X = X.to_numpy()

        for feature in X.T:
            res = {}
            vals = np.unique(np.array(feature))
            for val in vals:
                successes = np.mean(Y[feature == val])
                counters = np.sum(1 * feature == val) / feature.size
                res[val] = [successes, counters]

            self.data.append(res)

    def transform(self, X, a=1e-5, b=1e-5):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        param a: constant for counters, float
        param b: constant for counters, float
        returns: transformed objects, numpy-array, shape [n_objects, 3 * n_features]
        """
        res = np.zeros((X.shape[0], X.shape[1] * 3), dtype=self.dtype)
        X = X.to_numpy()
        for (index, feature) in enumerate(X.T):
            for i in range(feature.size):
                successes, counters = self.data[index][feature[i]]
                res[i][3 * index] = successes
                res[i][3 * index + 1] = counters
                res[i][3 * index + 2] = (successes + a) / (counters + b)

        return res

    def fit_transform(self, X, Y, a=1e-5, b=1e-5):
        self.fit(X, Y)
        return self.transform(X, a, b)

    def get_params(self, deep=True):
        return {"dtype": self.dtype}


def group_k_fold(size, n_splits=3, seed=1):
    idx = np.arange(size)
    np.random.seed(seed)
    idx = np.random.permutation(idx)
    n_ = size // n_splits
    for i in range(n_splits - 1):
        yield idx[i * n_: (i + 1) * n_], np.hstack((idx[:i * n_], idx[(i + 1) * n_:]))
    yield idx[(n_splits - 1) * n_:], idx[:(n_splits - 1) * n_]


class FoldCounters:

    def __init__(self, n_folds=3, dtype=np.float64):
        self.dtype = dtype
        self.n_folds = n_folds

    def fit(self, X, Y, seed=1):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: target for training objects, pandas-series, shape [n_objects,]
        param seed: random seed, int
        """
        self.folds = []
        self.data = []
        X = X.to_numpy()
        Y = Y.to_numpy()
        for folds in group_k_fold(X.shape[0], self.n_folds, seed):
            self.folds.append(folds)
        for fold in self.folds:
            X_1 = X[fold[1], :]
            Y_1 = Y[fold[1]]
            fold_data = []
            for feature in X_1.T:
                res = {}
                vals = np.unique(np.array(feature))
                for val in vals:
                    successes = np.mean(Y_1[feature == val])
                    counters = np.sum(1 * feature == val) / feature.size
                    res[val] = [successes, counters]
                fold_data.append(res)
            self.data.append(fold_data)

    def transform(self, X, a=1e-5, b=1e-5):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        param a: constant for counters, float
        param b: constant for counters, float
        returns: transformed objects, numpy-array, shape [n_objects, 3 * n_features]
        """
        res = np.zeros((X.shape[0], X.shape[1] * 3), dtype=self.dtype)
        X = X.to_numpy()
        for (fold_index, fold) in enumerate(self.folds):
            X_folded = X[fold[0], :]
            for (index, feature) in enumerate(X_folded.T):
                for i in range(feature.size):
                    successes = self.data[fold_index][index][feature[i]][0]
                    counters = self.data[fold_index][index][feature[i]][1]
                    res_folded = res[fold[0], :]
                    res_folded[i][3 * index] = successes
                    res_folded[i][3 * index + 1] = counters
                    res_folded[i][3 * index + 2] = (successes + a) / (counters + b)
                    res[fold[0], :] = res_folded
        return res

    def fit_transform(self, X, Y, a=1e-5, b=1e-5):
        self.fit(X, Y)
        return self.transform(X, a, b)


def weights(x, y):
    """
    param x: training set of one feature, numpy-array, shape [n_objects,]
    param y: target for training objects, numpy-array, shape [n_objects,]
    returns: optimal weights, numpy-array, shape [|x unique values|,]
    """

    encoder = MyOneHotEncoder()
    encoder.fit(x)
    X = encoder.transform(x)
    new = np.hstack((X, y[:, None]))
    w = []
    for i in range(X.shape[1]):
        w_1 = new[new[:, i] == 1][:, -1]
        w.append(np.mean(w_1))
    return np.array(w)
