from base import EnsembleTemplate
import numpy as np

from scipy.stats import kstest

class OUTRES(EnsembleTemplate):
    """
    Implementation of the algorithm OUTRES: Statistical Selection of Relevant Subspace
    Projections for Outlier Ranking. Authors: Emmanuel Müller, Matthias Schiffer and Thomas Seidl.

    Parameters
    ------------

    """

    def __init__(self, contamination=0.1, alpha=0.01):
        '''
        @brief Constructor of the class
        @param self
        @param contamination Percentage of outliers expected to be in the dataset
        @param alpha Parameter for the Kolmogorov Smirnov test to check if the
        data is uniformly distributed
        '''
        self.alpha = alpha

    def runMethod(self):
        '''
        @brief This function is the actual implementation of HICS
        '''

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
