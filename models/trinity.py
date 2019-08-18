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
    * num_iter: number of iterations to repeat the sampling process

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
        '''
        @brief Function that implements the distance based component
        @param self
        @return It returns the vector with the scores of the instances
        '''
        # Initialize the scores
        scores = np.array([0]*len(self.dataset)).astype(float)
        for i in range(self.num_iter):
            knn = KNN(n_neighbors=5, contamination=self.contamination)
            # Number in the interval [50, 1000]
            subsample_size = np.random.randint(50, 1001)
            sample = []
            if subsample_size>=len(self.dataset):
                sample = list(range(len(self.dataset)))
            else:
                # Take the sample and train the model
                sample = np.random.choice(len(self.dataset), size=subsample_size, replace=False)
            knn.fit(self.dataset[sample])
            # Update the score to compute the mean
            scores[sample]+=knn.decision_scores_
        # Return the mean
        return scores/self.num_iter

    def dependencyBased(self):
        '''
        @brief Function that implements the dependency based component
        @param self
        @return It returns the vector with the scores of the instances
        '''
        # Initialize the scores
        scores = np.array([0]*len(self.dataset)).astype(float)
        for i in range(self.num_iter):
            kernel_mahalanobis = KernelMahalanobis(contamination=self.contamination)
            # Number in the interval [50, 1000]
            subsample_size = np.random.randint(50, 1001)
            sample = []
            if subsample_size>=len(self.dataset):
                sample = list(range(len(self.dataset)))
            else:
                # Take the sample and train the model
                sample = np.random.choice(len(self.dataset), size=subsample_size, replace=False)
            kernel_mahalanobis.fit(self.dataset[sample])
            # Update the score to compute the mean
            scores[sample]+=kernel_mahalanobis.outlier_score
        # Return the mean
        return scores/self.num_iter

    def densityBased(self):
        '''
        @brief Function that implements the dependency based component
        @param self
        @return It returns the vector with the scores of the instances
        '''
        # Initialize the scores
        scores = np.array([0]*len(self.dataset)).astype(float)
        for i in range(self.num_iter):
            iforest = IForest(contamination=self.contamination, behaviour="new")
            # Number in the interval [50, 1000]
            subsample_size = np.random.randint(50, 1001)
            sample = []
            if subsample_size>=len(self.dataset):
                sample = list(range(len(self.dataset)))
            else:
                # Take the sample and train the model
                sample = np.random.choice(len(self.dataset), size=subsample_size, replace=False)
            iforest.fit(self.dataset[sample])
            # Update the score to compute the mean
            scores[sample]+=iforest.decision_scores_
        # Return the mean
        return scores/self.num_iter

    def runMethod(self):
        '''
        @brief This function is the actual implementation of TRINITY
        @param self
        '''
        # Distance module
        if self.verbose:
            print("Obtaining scores with the distance module")
        distance_based = self.distanceBased()
        # dependency module
        if self.verbose:
            print("Obtaining scores with the dependency module")
        dependency_based = self.dependencyBased()
        # Density module
        if self.verbose:
            print("Obtaining scores with the density module")
        density_based = self.densityBased()

        # Compute the mean of the three modules
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
