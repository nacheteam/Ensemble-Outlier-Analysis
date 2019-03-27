#!# -*- coding: utf-8 -*-
import numpy as np

class SOE1:
    '''
    Implements the SOE1 Ensemble method for Outlier Detection.
    The implementatio idea is picked from the paper A Unified Subspace Outlier
    Ensemble Framework for Outlier Detection written by Zengyou He, Shengchun Deng
    and Xiaofei Xu.

    Parameters
    ------------
    k: k most outlying objects to be extracted
    SS: set of subspaces of the data
    comb: combination/ensemble function
    '''

    def __init__(self,k,SS,comb,dataset,score_comb):
        '''
        @brief Function that initialises the SOE1 class
        @param self
        @param k Number of outliers wanted to be extracted
        @param SS list of subspaces of the characteristics given as lists of indexes
        in the dataset
        @param comb Combination function should take a bunch of histograms in a numpy
        array objetct and combine them into one single histogram.
        @param score_comb Combination function for the outlierness scores
        '''
        self.k = k
        self.SS = SS
        self.comb = comb
        self.dataset = dataset
        self.score_comb = score_comb

    def runMethod(self):
        '''
        @brief Function that executes the SOE1 method. The results are
        stored on the variable self.scores
        @param self
        '''
        dimension = len(self.dataset[0])
        histograms1D = []
        for i in range(dimension):
            subset = self.dataset[:,i]
            unique, counts = numpy.unique(subset, return_counts=True)
            histograms1D.append(zip(unique,counts))
        histograms1D = np.array(histograms1D)
        histogramsSS = []
        for subs in self.SS:
            histogramsSS.append(self.comb(histograms1D[subs]))
        histogramsSS = np.array(histogramsSS)
        self.scores = self.score_comb(np.append(histograms1D,histogramsSS))[:k]

    def obtainResults(self):
        '''
        @brief Function that, given the calculations donde by runMethod and the
        data stored with it, gives back some statistical information about the results.
        @param feature_names Names of the characteristics or features involved in
        the dataset
        '''
