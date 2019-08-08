from base import EnsembleTemplate
import numpy as np

from pyod.models.knn import KNN

class TRINITY(EnsembleTemplate):
    """
    Implementation of the algorithm TRINITY.
    Authors: Charu C. Aggarwal, Saket Sathe.

    Parameters
    ------------

    """

    def __init__(self, contamination=0.1, num_iter=100):
        '''
        @brief Constructor of the class
        @param self
        @param contamination Percentage of outliers expected to be in the dataset
        '''
        self.contamination=contamination

    def distanceBased(self):
        scores = np.array([0]*len(self.dataset))
        for i in range(self.num_iter):
            knn = KNN(n_neighbors=5, contamination=self.contamination)
            # Number in the interval [50, 1000]
            subsample_size = np.random.randint(50, 1001)
            sample = np.random.choice(len(self.dataset), size=subsample_size, replace=False)
            knn.fit(self.dataset[sample])
            scores[sample]+=knn.decision_scores_
        return scores/self.num_iter

    def runMethod(self):
        '''
        @brief This function is the actual implementation of TRINITY
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
