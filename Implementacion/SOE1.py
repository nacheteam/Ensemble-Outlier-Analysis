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

    def __init__(self,k,SS,comb):
        '''
        @brief Function that initialises the SOE1 class
        @param self
        '''

    def runMethod(self):
        '''
        @brief Function that executes the SOE1 method. The results are
        stored on the variable self.scores
        @param self
        '''

    def obtainResults(self):
        '''
        @brief Function that, given the calculations donde by runMethod and the
        data stored with it, gives back some statistical information about the results.
        @param feature_names Names of the characteristics or features involved in
        the dataset
        '''
