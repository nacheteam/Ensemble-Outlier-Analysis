#!# -*- coding: utf-8 -*-
from base import EnsembleTemplate

import numpy as np
from sklearn.preprocessing import scale
from scipy import stats
from scipy.spatial import distance_matrix

# For plotting
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class KernelMahalanobis(EnsembleTemplate):
    '''
    Implements the Kernel Mahalanobis Ensemble method for Outlier Detection.
    The implementation is picked from Aggarwal, Charu C., Sathe, Saket Outlier Ensembles

    Parameters
    ------------
    Parameter free
    '''

    def __init__(self, contamination=0.1):
        '''
        @brief Function that initialises the KernelMahalanobis class
        @param self
        @param contamination Proportion of outliers expected in the dataset, float
        between 0 and 1.
        '''
        self.contamination=0.1
        self.calculations_done=False

    def fit(self, dataset):
        '''
        Function to set the dataset and execute the algorithm
        '''
        self.dataset = np.matrix(dataset)
        self.runMethod()
        return self

    def runMethod(self):
        '''
        @brief Function that executes the Kernel Mahalanobis method. The results are
        stored on the variable self.scores
        @param self
        '''
        # Compute the S matrix of the algorithm
        S = np.dot(self.dataset, self.dataset.T)
        # Now we diagonalize it
        Q,delta_sq,Qt = np.linalg.svd(S)
        # Obtain delta as matrix
        delta = np.matrix(np.diag(np.sqrt(delta_sq)))
        Q = np.matrix(Q)
        # Compute de D' matrix and normalize it
        Dprime = np.dot(Q,delta)
        Dp_std = scale(Dprime, axis=1)
        # We compute its mean on the rows to compute the deviation as the score
        mean = Dp_std.mean(axis=0)
        self.outlier_score=[]
        # The score is the euclidean distance to the mean
        for i in range(len(Dp_std)):
            self.outlier_score.append(np.linalg.norm(mean-Dp_std[i])**2)
        self.outlier_score = np.array(self.outlier_score)
        self.calculations_done=True

    def getOutliersBN(self, noutliers):
        '''
        @brief Function that gives the noutliers most outlying instances
        @param noutliers Number of outliers to return
        @return It returns the indexes of the noutliers most outlying instances
        '''
        assert self.calculations_done, ("The method needs to be executed before obtaining the outliers")
        if noutliers<len(self.dataset):
            return self.outlier_score.argsort()[-noutliers:][::-1]
        return np.array(list(range(len(self.dataset))))

    def getOutliers(self):
        '''
        @brief Function that gives the outliers based on the contamination parameter
        '''
        assert self.calculations_done, ("The method needs to be executed before obtaining the outliers")
        num_out = int(self.contamination*len(self.dataset))
        return self.outlier_score.argsort()[-num_out:][::-1]
