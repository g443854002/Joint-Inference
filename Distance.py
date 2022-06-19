from abcpy.distances import Distance
import copy
import math
import numpy as np
from glmnet import LogitNet
from abc import ABCMeta, abstractmethod
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression,RidgeCV,LogisticRegressionCV
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from yaglm.GlmTuned import GlmCV
from yaglm.config.penalty import Lasso
from yaglm.config.flavor import Adaptive

class Absolute(Distance):
    """
    This class implements the Euclidean distance between two vectors.

    The maximum value of the distance is np.inf.
    """

    def __init__(self, statistics):
        self.statistics_calc = statistics

        # Since the observations do always stay the same, we can save the
        #  summary statistics of them and not recalculate it each time
        self.s1 = None
        self.data_set = None
        self.dataSame = False

    def distance(self, d1, d2):
        """Calculates the distance between two datasets.

        Parameters
        ----------
        d1, d2: list
            A list, containing a list describing the data set
        """

        if not isinstance(d1, list):
            raise TypeError('Data is not of allowed types')
        if not isinstance(d2, list):
            raise TypeError('Data is not of allowed types')

        # Check whether d1 is same as self.data_set
        if self.data_set is not None:
            if len(np.array(d1[0]).reshape(-1, )) == 1:
                self.data_set == d1
            else:
                self.dataSame = all([(np.array(self.data_set[i]) == np.array(d1[i])).all() for i in range(len(d1))])

        # Extract summary statistics from the dataset
        if (self.s1 is None or self.dataSame is False):
            self.s1 = self.statistics_calc.statistics(d1)
            self.data_set = d1

        s2 = self.statistics_calc.statistics(d2)

        # compute distance between the statistics
        dist = np.zeros(shape=(self.s1.shape[0], s2.shape[0]))
        for ind1 in range(0, self.s1.shape[0]):
            for ind2 in range(0, s2.shape[0]):
                dist[ind1, ind2] = sum(abs(self.s1[ind1, :] - s2[ind2, :]))

        return dist.mean()

    def dist_max(self):
        return np.inf
    
class Divergence(Distance, metaclass=ABCMeta):
    """This is an abstract class which subclasses Distance, and is used as a parent class for all divergence
    estimators; more specifically, it is used for all Distances which compare the empirical distribution of simulations
    and observations."""

    @abstractmethod
    def _estimate_always_positive(self):
        """This returns whether the implemented divergence always returns positive values or not. In fact, some 
        estimators may return negative values, which may break some inference algorithms"""
        raise NotImplementedError

class PenLogReg(Divergence):
    """
    This class implements a distance measure based on the classification accuracy.
    The classification accuracy is calculated between two dataset d1 and d2 using
    lasso penalized logistics regression and return it as a distance. The adaptive lasso
    penalized logistic regression is done using glmnet package of Friedman et. al.
    [2][3]. While computing the distance, the algorithm automatically chooses
    the most relevant summary statistics as explained in Gutmann et. al. [1].
    The maximum value of the distance is 1.0.
    [1] Gutmann, M. U., Dutta, R., Kaski, S., & Corander, J. (2018). Likelihood-free inference via classification.
    Statistics and Computing, 28(2), 411-425.
    [2] Friedman, J., Hastie, T., and Tibshirani, R. (2010). Regularization
    paths for generalized linear models via coordinate descent. Journal of Statistical
    Software, 33(1), 1–22.
    [3] Hui Zou. The adaptive lasso and its oracle properties. Journal of the American statistical association, 101(476): 1418–1429, 2006.11

    Parameters
    ----------
    statistics_calc : abcpy.statistics.Statistics
        Statistics extractor object that conforms to the Statistics class.
    """

    def __init__(self, statistics_calc):
        super(PenLogReg, self).__init__(statistics_calc)

        self.n_folds = 3 # for cross validation in PenLogReg

    def distance(self, d1, d2):
        """Calculates the distance between two datasets.
        Parameters
        ----------
        d1: Python list
            Contains n1 data points.
        d2: Python list
            Contains n2 data points.
        Returns
        -------
        numpy.float
            The distance between the two input data sets.
        """
        s1, s2 = self._calculate_summary_stat(d1, d2)
        self.n_simulate = s1.shape[0]

        if not s2.shape[0] == self.n_simulate:
            raise RuntimeError("The number of simulations in the two data sets should be the same in order for "
                               "the classification accuracy implemented in PenLogReg to be a proper distance. Please "
                               "check that `n_samples` in the `sample()` method for the sampler is equal to "
                               "the number of datasets in the observations.")
        
        

        # Create matrix of features and labels for classification
        # Observed data denoted as 0, simulated data as 1.
        training_set_features = np.concatenate((s1,s2), axis=0)

        label_s1 = np.zeros(shape=(len(s1), 1))
        label_s2 = np.ones(shape=(len(s2), 1))
        training_set_labels = np.concatenate((label_s1, label_s2), axis=0).ravel()
        # Random shuffle the data to make sure model remains general and avoid overfit.
        training_set_features,training_set_labels = shuffle(training_set_features,training_set_labels)
        # Fit logistic regression with L-2 penalty        
        lr = LogitNet(alpha=1, n_splits=self.n_folds,standardize=False).fit(training_set_features, training_set_labels)
        # Extract the coefficients from the logistic regression with L-2 penalty
        x_coefs = lr.coef_.flatten()
        # Use the coefficient calculate above to further calculate weights for adaptive lasso
        weights = 1/(np.abs(x_coefs)+0.00001)
        # Fit adaptive lasso with initial weights calculated above
        m = LogitNet(alpha=1,n_splits=self.n_folds,standardize=False)  
        m = m.fit(training_set_features, training_set_labels,relative_penalties=weights)
        # # # # Extract the cross validated accuracy rate from the adaptive lasso.
        distance = m.cv_mean_score_[np.where(m.lambda_path_ == m.lambda_max_)[0][0]]

        return distance

    def dist_max(self):
        """
        Returns
        -------
        numpy.float
            The maximal possible value of the desired distance function.
        """
        return 1.0

    def _estimate_always_positive(self):
        return True
    

