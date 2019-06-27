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
    iter_number: number of iterations for the subsampling loop.
    '''

    def __init__(self,iter_number):
        '''
        @brief Function that initialises the KernelMahalanobis class
        @param self
        @param iter_number Number of times subsampling is going to be applied
        '''
        self.niter=iter_number
        self.calculations_done=False

    def runMethod(self):
        '''
        @brief Function that executes the Kernel Mahalanobis method. The results are
        stored on the variable self.scores
        @param self
        '''
        S = np.dot(self.dataset, self.dataset.T)
        Q,delta_sq,Qt = np.linalg.svd(S)
        delta = np.matrix(np.diag(np.sqrt(delta_sq)))
        Q = np.matrix(Q)
        Dprime = np.dot(Q,delta)
        Dp_std = scale(Dprime, axis=1)
        mean = Dp_std.mean(axis=0)
        self.outlier_score=[]
        for i in range(len(Dp_std)):
            self.outlier_score.append(np.linalg.norm(mean-Dp_std[i])**2)
        self.outlier_score = np.array(self.outlier_score)
        self.calculations_done=True

    def getOutliersBN(self, noutliers):
        if noutliers<len(self.dataset):
            return self.outlier_score.argsort()[-noutliers:][::-1]
        return np.array(list(range(len(self.dataset))))

    """
    def runMethod(self):
        '''
        @brief Function that executes the Kernel Mahalanobis method. The results are
        stored on the variable self.scores
        @param self
        '''
        total_scores = None
        empty_scores = True
        # For each iteration
        for i in range(self.niter):
            print("Iteration "  + str(i+1) + "/" + str(self.niter))
            # Pull a random integer in (min{50,n},min{1000,n})
            s = np.random.randint(low=min([50,self.datasize]),high=min([1000,self.datasize]))
            # Create the subsample with size s
            subsample_indices = np.random.randint(self.datasize,size=s)
            subsample = self.dataset[subsample_indices,:]
            # Create the similarity matrix with the subsampled data
            sim_matrix = distance_matrix(subsample,subsample)
            # Use SVD to decompose S = QΔ^2Q^t
            Q,delta_sq,Qt = np.linalg.svd(sim_matrix)
            delta = np.diag(np.sqrt(delta_sq))
            Q = np.matrix(Q)
            delta_sq = np.matrix(delta_sq)
            # Obtain non-zero indices of Q vectors and obtain the correspondent Qk and Δk
            non_zero_ind = list(set(np.nonzero(Q)[1]))
            Qk=Q[:,non_zero_ind]
            deltak = delta[:,non_zero_ind]
            # Build the similarity matrix of the points out of the sample
            out_indices = list(set(range(self.datasize)).difference(set(subsample_indices)))
            out_subsample = self.dataset[out_indices,:]
            out_sim_matrix = distance_matrix(self.dataset,subsample)
            # Build the total embedding and standardize
            D = np.dot(out_sim_matrix,np.dot(Qk, np.linalg.inv(deltak)))
            D_stand = stats.zscore(D)
            mean = D_stand.mean(0).reshape(-1)
            # Compute the score
            dimensions = len(D_stand[0])
            score=[]
            for j in range(len(D_stand)):
                score.append(np.linalg.norm(D_stand[j]-mean)/dimensions)
            if empty_scores:
                empty_scores=False
                total_scores=np.array(score)/self.niter
            else:
                total_scores=total_scores+np.array(score)/self.niter
        self.scores = np.array(total_scores)
        self.calculations_done=True
    """
