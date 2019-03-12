'''
Author: Apoorv Agnihotri

This file implements KNN for Classification and Regression
tasks
'''

import distance
import distance as DMetric
import numpy as np
from utils import mode

class KNN:
    """You don't want to instantiate this
    object directly, use a derived class.
    """
    def __init__(self,
                 k,
                 dist_metric):
        self.k = k
        self.dmetric = self._make_metric_class(dist_metric)
        
    def train(self, X, y):
        """Use this function to train"""
        self.X = X
        self.y = y

    def _make_metric_class(self, dist_metric):
        """This function helps in modifying the distance
        object on the fly, given classification or regression
        """
        if dist_metric=='euclidian':
            return DMetric.Euclidian()

        elif dist_metric=='manhattan':
            return DMetric.Manhattan()
        
        elif dist_metric=='cosine':
            return DMetric.Cosine()
    
    def predict(self, X_test):
        # for each row in df, apply self._predict_one
        y_preds = []
        for point in range(X_test.shape[0]):
            point_dist = []
            point_dists = self.dmetric.calc_dist(self.X, X_test[point, :])
            point_dist = np.array(point_dists)
            minimizer_ix = point_dist.argsort()[:(self.k)][::-1]
            y_pred = self.predict_reducer(minimizer_ix)
            y_preds.append(y_pred)
        return np.squeeze(np.array(y_preds))


class KNNClassifier(KNN):
    """Instantiate this class for a KNN for classification tasks."""
    def __init__(self,
                 k = 2,
                 dist_metric = 'euclidian'):
        super().__init__(k = k,
                         dist_metric = dist_metric)

    def predict_reducer(self, minimizer_ix):
        '''returns the average y's for the closest X's'''
        return mode(self.y[minimizer_ix])[0]
    

class KNNRegression(KNN):
    """Instantiate this class for a KNN for Regression tasks."""
    def __init__(self,
                 k = 2,
                 dist_metric = 'euclidian'):
        super().__init__(k = k,
                         dist_metric = dist_metric)

    def predict_reducer(self, minimizer_ix):
        '''returns the average y's for the closest X's'''
        return np.average(self.y[minimizer_ix])
