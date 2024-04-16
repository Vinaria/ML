import os

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import Pipeline
import numpy as np


def ver_centrate(pic):
    x = pic.shape[0]
    # pic2 = np.zeros(pic.shape)
    # pic2[pic == pic.min()] = 1
    m = pic.min()

    for i, row in enumerate(pic):
        if m in row:
            upper = i
            break

    for i, row in enumerate(np.flipud(pic)):
        if m in row:
            lower = i
            break

    x1 = upper
    x2 = x - lower
    delta = np.abs(x // 2 - (x1 + x2) // 2)

    if upper > lower:
        pic3 = np.concatenate((pic, np.full((delta, 256), 20)), axis=0)[delta: 256 + delta, :]
    else:
        pic3 = np.concatenate((np.full((delta, 256), 20), pic), axis=0)[0: 256, :]

    return pic3


class PotentialTransformer:
    """
    A potential transformer.

    This class is used to convert the potential's 2d matrix to 1d vector of features.
    """

    def fit(self, x, y):

        return self

    def fit_transform(self, x, y):
        """
        Build the transformer on the training set and return the transformed dataset (1d vectors).
        :param x: list of potential's 2d matrices
        :param y: target values (can be ignored)
        :return: transformed potentials (list of 1d vectors)
        """
        self.fit(x, y)

        return self.transform(x)

    def transform(self, x):
        """
        Transform the list of potential's 2d matrices with the trained transformer.
        :param x: list of potential's 2d matrices
        :return: transformed potentials (list of 1d vectors)
        """
        for i in range(x.shape[0]):
            new_pic = ver_centrate(ver_centrate(x[i]).T).T
            x[i] = new_pic
        return x.reshape((x.shape[0], -1))


def load_dataset(data_dir):
    """
    Read potential dataset.

    This function reads dataset stored in the folder and returns three lists
    :param data_dir: the path to the potential dataset
    :return:
    files -- the list of file names
    np.array(X) -- the list of potential matrices (in the same order as in files)
    np.array(Y) -- the list of target value (in the same order as in files)
    """
    files, X, Y = [], [], []
    for file in sorted(os.listdir(data_dir)):
        potential = np.load(os.path.join(data_dir, file))
        files.append(file)
        X.append(potential["data"])
        Y.append(potential["target"])
    return files, np.array(X), np.array(Y)


def train_model_and_predict(train_dir, test_dir):

    #      tr = tracker.SummaryTracker()

    #      _, X_train, Y_train = load_dataset(train_dir)

    #     tr.print_diff()

    #     it's suggested to modify only the following line of this function
    #     X_train = np.concatenate((X_train, [np.flip(obj) for obj in X_train], [np.rot90(obj) for obj in X_train]))
    #     Y_train = np.concatenate((Y_train, Y_train, Y_train))
    #     regressor = Pipeline([
    #         ('transfromer', PotentialTransformer()),
    #         ('model', ExtraTreesRegressor(n_estimators=800, max_depth=9, max_features="sqrt"))
    #     ])
    #     regressor.fit(X_train, Y_train)
    #     tr.print_diff()
    #      del X_train, Y_train

    #     test_files, X_test, _ = load_dataset(test_dir)
    #     predictions = regressor.predict(X_test)
    #     tr.print_diff()
    #     del X_test
    #     tr.print_diff()

    _, X_train, Y_train = load_dataset(train_dir)
    test_files, X_test, _ = load_dataset(test_dir)
    # it's suggested to modify only the following line of this function
    rfr = ExtraTreesRegressor(n_estimators=800, max_depth=9, max_features="sqrt", warm_start=True)
    transformer = PotentialTransformer()

    X_train_transformed = transformer.fit_transform(X_train, Y_train)
    rfr.fit(X_train_transformed, Y_train)
    # rfr.n_estimators += 800

    for pot in range(X_train.shape[0]):
        X_train[pot] = np.rot90(X_train[pot], 1)

    X_train_transformed = transformer.fit_transform(X_train, Y_train)
    rfr.fit(X_train_transformed, Y_train)
    # rfr.n_estimators += 800

    for pot in range(X_train.shape[0]):
        X_train[pot] = np.flip(X_train[pot], axis=0)

    X_train_transformed = transformer.fit_transform(X_train, Y_train)
    rfr.fit(X_train_transformed, Y_train)

    del X_train_transformed
    del X_train
    X_test = transformer.transform(X_test)
    predictions = rfr.predict(X_test)
    return {file: value for file, value in zip(test_files, predictions)}
