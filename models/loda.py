from base import EnsembleTemplate
import numpy as np

class LODA(EnsembleTemplate):
    """
    Implementation of the algorithm LODA: Lightweight on-line detector of anomalies
    Authors: Tomas Pevny

    Parameters
    ------------
    * k: number of projections and histograms to do
    * n_bins: number of bins of the histogram
    """

    def __init__(self, contamination=0.1, n_bins=25, k=500):
        self.contamination = contamination
        self.n_bins = n_bins
        self.k = k

    def getRandomProjections(self, dimension):
        non_neg = int(np.ceil(np.sqrt(dimension)))
        projections = []
        for i in range(self.k):
            ind = np.random.choice(dimension, replace=False, size=non_neg)
            proj = np.zeros(dimension)
            proj[ind]=np.random.normal(size=non_neg)
            projections.append(proj)
        return projections

    def getBin(self, hist_limits, value):
        bin=-1
        for i in range(len(hist_limits)):
            if value<hist_limits[i]:
                bin=i
                break
        return bin

    def runMethod(self):
        random_projections = self.getRandomProjections(len(self.dataset[0]))
        histograms = [[]]*self.k
        limits = histograms = [[]]*self.k
        Z = [[]]*self.k
        for j in range(len(self.dataset)):
            for i in range(self.k):
                Z[i].append(np.dot(self.dataset[j].T, random_projections[i]))
        for i in range(self.k):
            histograms[i]=np.histogram(Z[i], bins = self.n_bins)

        self.outlier_score = np.array([0]*len(self.dataset))
        for i in range(len(self.dataset)):
            prob = []
            for j in range(self.k):
                z = np.dot(self.dataset[i].T, random_projections[j])
                bin = self.getBin(histograms[j][1], z)-1
                prob.append(histograms[j][0][bin]/np.sum(histograms[j][0]))
            prob = np.array(prob)
            self.outlier_score[i] = -np.sum(np.log(prob))/self.k
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
