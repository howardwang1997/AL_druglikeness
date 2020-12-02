from __future__ import division

import collections
import copy
import os

import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel, linear_kernel
from sklearn.neighbors import kneighbors_graph
from sklearn.utils.multiclass import unique_labels

from alipy.query_strategy.base import BaseIndexQuery
from alipy.utils.ace_warnings import *
from alipy.utils.misc import nsmallestarg, randperm, nlargestarg

__all__ = ['QueryInstanceDistruibution',
           'QueryInstanceResidue']

class QueryInstanceDistribution(BaseIndexQuery):
    """Distribution query strategy.
    The implement of uncertainty measure includes:
    1. Distance
    2. Cohn
    3. Variance 

    The above measures need the probabilistic output of the model.
    There are 3 ways to select instances in the data set.
    1. use select_by_prediction by providing the probabilistic prediction
       matrix of your own model, shape [n_samples, n_classes].

    Parameters
    ----------
    X: 2D array, optional (default=None)
        Feature matrix of the whole dataset. It is a reference which will not use additional memory.

    y: array-like, optional (default=None)
        Label matrix of the whole dataset. It is a reference which will not use additional memory.

    measure: str, optional (default='entropy')
        measurement to calculate uncertainty, should be one of
        ['least_confident', 'margin', 'entropy', 'distance_to_boundary']
        --'least_confident' x* = argmax 1-P(y_hat|x) ,where y_hat = argmax P(yi|x)
        --'margin' x* = argmax P(y_hat1|x) - P(y_hat2|x), where y_hat1 and y_hat2 are the first and second
            most probable class labels under the model, respectively.
        --'entropy' x* = argmax -sum(P(yi|x)logP(yi|x))
        --'distance_to_boundary' Only available in binary classification, x* = argmin |f(x)|,
            your model should have 'decision_function' method which will return a 1d array.

    """

    def __init__(self, X=None, y=None, measure='distance', a=1, b=1, regressor=None):
        if measure not in ['distance', 'variance', 'nearest_neighbor']:
            raise ValueError("measure must be one of ['distance', 'variance', 'nearest_neighbor']")
        self.measure = measure
        self.a, self.b = a, b
        self.X, self.y = X,y
        super(QueryInstanceDistribution, self).__init__(X, y)

    def select(self, *args, **kwargs):
        pass

    def select_by_prediction(self, unlabel_index, predict, labels,
                                 batch_size=1, X_lab=None, X_unlab=None, previous=None):
        """Select indexes from the unlabel_index for querying.

        Parameters
        ----------
        unlabel_index: {list, np.ndarray, IndexCollection}
            The indexes of unlabeled samples. Should be one-to-one
            correspondence to the prediction matrix.

        predict_unlab: 1d array, shape [n_samples_unlab, ]
            The prediction of the unlabeled set.

        labels: 1d array, shape [n_samples_labeled, ]
            The labels for the labeled set.

        batch_size: int, optional (default=1)
            Selection batch size.

        X_lab: 2d array, shape [n_samples_labeled, n_features]
            The labeled dataset.

        X_unlab: 2d array, shape [n_samples_unlab, n_features]
            The unlabeled dataset.

        Returns
        -------
        selected_idx: list
            The selected indexes which is a subset of unlabel_index.
        """
        assert (batch_size > 0)
        assert (isinstance(unlabel_index, collections.Iterable))
        unlabel_index = np.asarray(unlabel_index)
        if len(unlabel_index) <= batch_size:
            return unlabel_index

        pu = np.asarray(predict)  # predict value of unlabeled set
        spu = np.shape(pu)  # shape of pu
        lb = np.asarray(labels)  # predict value of labeled set
        slb = np.shape(lb)  # shape of lb

        if self.measure == 'distance':
            # calc distance
            a,b = self.a, self.b  # nonlinearity parameter, a=b=0 for exploration, a=b=1 for nonlinearity sampling
            distance = []
            X_lab, X_unlab = np.asarray(X_lab), np.asarray(X_unlab)
            if previous is None: dvar = np.array([0] * len(X_unlab))
            else:
                from scipy import integrate
                var = np.asarray([np.var(sample) for sample in previous.T])
                med_var = np.var(np.asarray([np.median(sample) for sample in previous.T]))
                dvar = var / med_var
            for i in range(len(unlabel_index)):
                dx = min([np.sqrt(np.sum((X_unlab[i] - lab_sample) ** 2)) for lab_sample in X_lab])
                dy = np.average([np.abs(lab - pu[i]) / (np.max(lb) - np.min(lb)) for lab in lb])
                dtot = dx * (1 + dy) ** a * (1 + dvar[i]) ** b
                if i < 10:
                    print('\n')
                    print(dx)
                    print(dy)
                    print(dvar[i])
                distance.append(dtot)
            return unlabel_index[nlargestarg(distance, batch_size)]

        elif self.measure == 'variance':
            # calc variance
            var = [np.var(np.hstack((sample,lb))) for sample in pu]
            return unlabel_index[nlargestarg(var, batch_size)]

        elif self.measure == 'nearest_neighbor':
            # calc distance to nearest neighbor of prediction
            nn = []
            for i in range(len(unlabel_index)):
                dy = min([np.abs(lab - pu[i]) / (np.max(lb) - np.min(lb)) for lab in lb])
                nn.append(dy)
            return unlabel_index[nlargestarg(nn, batch_size)]



class QueryInstanceResidueRegressor(BaseIndexQuery):
    """Regressor query strategy.
    The implement of criterion measure includes:
    1. Residue

    The above measures need a neural network regressor.
    There is 1 way to select instances in the data set.
    1. use select_by_prediction by providing the regressor model

    Parameters
    ----------
    X: 2D array, optional (default=None)
        Feature matrix of the whole dataset. It is a reference which will not use additional memory.

    y: array-like, optional (default=None)
        Label matrix of the whole dataset. It is a reference which will not use additional memory.

    distance: string, optional (default='linear')
        'linear', 'absolute', 'square'. It is the method that residue is measured.

    """

    def __init__(self, X=None, y=None, distance='linear', a=1, b=1, regressor=None):
        self.measure = 'residue'
        self.a, self.b = a, b
        self.X, self.y = X,y
        self.distance = distance
        if not regressor:
            from sklearn.neural_network import MLPRegressor
            if len(y) < 120: hidden = int(len(y)/2)
            self.regressor = MLPRegressor(hidden_layer_sizes=(100,100,100), max_iter=1000)
        super(QueryInstanceResidueRegressor, self).__init__(X, y)

    def select(self, *args, **kwargs):
        pass

    def select_by_prediction(self, unlabel_index, predict, labels,
                                 batch_size=1, X_lab=None, X_unlab=None, previous=None):
        """Select indexes from the unlabel_index for querying.

        Parameters
        ----------
        unlabel_index: {list, np.ndarray, IndexCollection}
            The indexes of unlabeled samples. Should be one-to-one
            correspondence to the prediction matrix.

        predict_lab: 1d array, shape [n_samples_unlab, ]
            The prediction of the labeled set.

        labels: 1d array, shape [n_samples_labeled, ]
            The labels for the labeled set.

        batch_size: int, optional (default=1)
            Selection batch size.

        X_lab: 2d array, shape [n_samples_labeled, n_features]
            The labeled dataset.

        X_unlab: 2d array, shape [n_samples_unlab, n_features]
            The unlabeled dataset.

        Returns
        -------
        selected_idx: list
            The selected indexes which is a subset of unlabel_index.
        """
        assert (batch_size > 0)
        assert (isinstance(unlabel_index, collections.Iterable))
        unlabel_index = np.asarray(unlabel_index)
        if len(unlabel_index) <= batch_size:
            return unlabel_index

        pl = np.asarray(predict)  # predict value of unlabeled set
        spl = np.shape(pl)  # shape of pu
        lb = np.asarray(labels)  # predict value of labeled set
        slb = np.shape(lb)  # shape of lb
        res = pl - lb
        # if self.distance == 'linear': res = res
        if self.distance == 'absolute': res = np.abs(res)
        elif self.distance == 'square': res = res ** 2
        elif self.distance != 'linear': raise ValueError("variable 'distance' must be one of "
                                                         "'linear', 'absolute', 'square'")

        # predict the residue

        # from sklearn.preprocessing import StandardScaler
        # ss = StandardScaler()
        # X_lab = ss.fit_transform(X_lab)
        self.regressor.fit(X_lab, res)
        residue = self.regressor.predict(X_unlab)
        residue = np.abs(residue)

        return unlabel_index[nlargestarg(residue, batch_size)]


