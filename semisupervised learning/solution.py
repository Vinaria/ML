import numpy as np

import sklearn
from sklearn.cluster import KMeans


class KMeansClassifier(sklearn.base.BaseEstimator):
    def __init__(self, n_clusters):
        '''
        :param int n_clusters: Число кластеров которых нужно выделить в обучающей выборке с помощью алгоритма кластеризации
        '''
        super().__init__()
        self.n_clusters = n_clusters
        self.kmeans_clusterizator = KMeans(self.n_clusters)
        self.mapping = None

        # Ваш код здесь:＼(º □ º l|l)/

    def fit(self, data, labels):
        '''
            Функция обучает кластеризатор KMeans с заданным числом кластеров, а затем с помощью
        self._best_fit_classification восстанавливает разметку объектов

        :param np.ndarray data: Непустой двумерный массив векторов-признаков объектов обучающей выборки
        :param np.ndarray labels: Непустой одномерный массив. Разметка обучающей выборки. Неразмеченные объекты имеют метку -1.
            Размеченные объекты могут иметь произвольную неотрицательную метку. Существует хотя бы один размеченный объект
        :return KMeansClassifier
        '''
        self.kmeans_clusterizator.fit(data)
        cluster_labels = self.kmeans_clusterizator.predict(data)

        mapping, predicted_labels = self._best_fit_classification(cluster_labels, labels)

        self.mapping = mapping

        return self

    def predict(self, data):
        '''
        Функция выполняет предсказание меток класса для объектов, поданных на вход. Предсказание происходит в два этапа
            1. Определение меток кластеров для новых объектов
            2. Преобразование меток кластеров в метки классов с помощью выученного преобразования

        :param np.ndarray data: Непустой двумерный массив векторов-признаков объектов
        :return np.ndarray: Предсказанные метки класса
        '''
        cluster_labels = self.kmeans_clusterizator.predict(data)
        predictions = self.mapping[cluster_labels]

        return predictions

    def _best_fit_classification(self, cluster_labels, true_labels):
        '''
        :param np.ndarray cluster_labels: Непустой одномерный массив. Предсказанные метки кластеров.
            Содержит элементы в диапазоне [0, ..., n_clusters - 1]
        :param np.ndarray true_labels: Непустой одномерный массив. Частичная разметка выборки.
            Неразмеченные объекты имеют метку -1. Размеченные объекты могут иметь произвольную неотрицательную метку.
            Существует хотя бы один размеченный объект
        :return
            np.ndarray mapping: Соответствие между номерами кластеров и номерами классов в выборке,
                то есть mapping[idx] -- номер класса для кластера idx
            np.ndarray predicted_labels: Предсказанные в соответствии с mapping метки объектов

            Соответствие между номером кластера и меткой класса определяется как номер класса с максимальным числом объектов
        внутри этого кластера.
            * Если есть несколько классов с числом объектов, равным максимальному, то выбирается метка с наименьшим номером.
            * Если кластер не содержит размеченных объектов, то выбирается номер класса с максимальным числом элементов в выборке.
            * Если же и таких классов несколько, то также выбирается класс с наименьшим номером
        '''
        cluster_labels = np.array(cluster_labels)
        true_labels = np.array(true_labels)

        unique_classes, classes_counts = np.unique(true_labels[true_labels != -1], return_counts=True)
        biggest_class = unique_classes[classes_counts == classes_counts.max()][0]

        unique_clusters = np.unique(cluster_labels)
        mapping = np.empty(self.n_clusters)

        for i in range(self.n_clusters):
            classes_in_cluster = true_labels[cluster_labels == i]
            classes_in_cluster = classes_in_cluster[classes_in_cluster != -1]

            if not classes_in_cluster.size:
                mapping[i] = biggest_class
            else:
                unique, counts = np.unique(classes_in_cluster, return_counts=True)
                best_classes = unique[counts == counts.max()]
                mapping[i] = best_classes[0]

        predicted_labels = mapping[cluster_labels]

        return mapping, predicted_labels
