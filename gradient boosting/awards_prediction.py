from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from numpy import ndarray

"""
 Внимание!
 В проверяющей системе имеется проблема с catboost.
 При использовании этой библиотеки, в скрипте с решением необходимо инициализировать метод с использованием `train_dir` как показано тут:
 CatBoostRegressor(train_dir='/tmp/catboost_info')
"""


def train_model_and_predict(train_file: str, test_file: str) -> ndarray:
    """
    This function reads dataset stored in the folder, trains predictor and returns predictions.
    :param train_file: the path to the training dataset
    :param test_file: the path to the testing dataset
    :return: predictions for the test file in the order of the file lines (ndarray of shape (n_samples,))
    """

    df_train = pd.read_json(train_file, lines=True)
    df_test = pd.read_json(test_file, lines=True)

    # remove categorical variables

    vectorizer = CountVectorizer()

    y_train = df_train["awards"]
    del df_train["awards"]

    del df_train["keywords"]
    del df_test["keywords"]

    colnames = ['genres', 'directors', 'filming_locations']
    cat_names = ['actor_0_gender', 'actor_1_gender', 'actor_2_gender']

    cat_indices = df_train.columns.get_indexer(cat_names)
    df_train.loc[:, cat_names] = df_train.loc[:, cat_names].astype("category")
    df_test.loc[:, cat_names] = df_test.loc[:, cat_names].astype("category")

    for col in colnames:
        df_train[col] = df_train[col].apply(
            lambda x: x if (isinstance(x, str) and x.lower() == 'unknown') else ','.join(x)
        )
        df_test[col] = df_test[col].apply(
            lambda x: x if (isinstance(x, str) and x.lower() == 'unknown') else ','.join(x)
        )

        vect = CountVectorizer(token_pattern=r'[a-z ]+')
        train_vect = vect.fit_transform(df_train[col]).toarray()
        test_vect = vect.transform(df_test[col]).toarray()
        cols = [elem + '_' + col for elem in vect.get_feature_names_out()]

        del df_train[col]
        del df_test[col]

        df_train = pd.concat([df_train, pd.DataFrame(train_vect, columns=cols)], axis=1)
        df_test = pd.concat([df_test, pd.DataFrame(test_vect, columns=cols)], axis=1)

    regressor = CatBoostRegressor(
        max_depth=5,
        n_estimators=2013,
        learning_rate=0.01230817,
        verbose=0,
        train_dir='/tmp/catboost_info'
    )

    regressor.fit(
        df_train,
        y_train,
        cat_features=cat_names
    )

    return regressor.predict(df_test)
