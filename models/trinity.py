from base import EnsembleTemplate
import numpy as np

from pyod.models.knn import KNN
from KernelMahalanobis import KernelMahalanobis
from pyod.models.iforest import IForest

class TRINITY(EnsembleTemplate):
    """
    Implementation of the algorithm TRINITY.
    Authors: Charu C. Aggarwal, Saket Sathe.

    Parameters
    ------------

    """

    def __init__(self, contamination=0.1, num_iter=100, verbose=False):
        '''
        @brief Constructor of the class
        @param self
        @param contamination Percentage of outliers expected to be in the dataset
        '''
        self.contamination=contamination
        self.num_iter = num_iter
        self.verbose = verbose

    def distanceBased(self):
        scores = np.array([0]*len(self.dataset)).astype(float)
        for i in range(self.num_iter):
            knn = KNN(n_neighbors=5, contamination=self.contamination)
            # Number in the interval [50, 1000]
            subsample_size = np.random.randint(50, 1001)
            sample = np.random.choice(len(self.dataset), size=subsample_size, replace=False)
            knn.fit(self.dataset[sample])
            scores[sample]+=knn.decision_scores_
        return scores/self.num_iter

    def dependencyBased(self):
        scores = np.array([0]*len(self.dataset)).astype(float)
        for i in range(self.num_iter):
            kernel_mahalanobis = KernelMahalanobis(contamination=self.contamination)
            # Number in the interval [50, 1000]
            subsample_size = np.random.randint(50, 1001)
            sample = np.random.choice(len(self.dataset), size=subsample_size, replace=False)
            kernel_mahalanobis.fit(self.dataset[sample])
            scores[sample]+=kernel_mahalanobis.outlier_score
        return scores/self.num_iter

    def densityBased(self):
        scores = np.array([0]*len(self.dataset)).astype(float)
        for i in range(self.num_iter):
            iforest = IForest(contamination=self.contamination)
            # Number in the interval [50, 1000]
            subsample_size = np.random.randint(50, 1001)
            sample = np.random.choice(len(self.dataset), size=subsample_size, replace=False)
            iforest.fit(self.dataset[sample])
            scores[sample]+=iforest.decision_scores_
        return scores/self.num_iter

    def runMethod(self):
        '''
        @brief This function is the actual implementation of TRINITY
        '''
        if self.verbose:
            print("Obtaining scores with the distance module")
        distance_based = self.distanceBased()
        if self.verbose:
            print("Obtaining scores with the dependency module")
        dependency_based = self.dependencyBased()
        if self.verbose:
            print("Obtaining scores with the density module")
        density_based = self.densityBased()
        self.outlier_score=(distance_based + dependency_based + density_based)/3
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
