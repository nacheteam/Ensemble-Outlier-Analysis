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
        '''
        @brief Constructor of the class
        @param self
        @param contamination Parameter indicating the percentage of anomalies expected
        in the dataset
        @param n_bins Number of bins of the histograms
        @param k Number of histograms and projections to compute
        '''
        self.contamination = contamination
        self.n_bins = n_bins
        self.k = k

    def getRandomProjections(self, dimension):
        '''
        @brief Function that computes and returns the random projections
        @param self
        @param dimension Dimensionality of the dataset (int)
        @return It returns a list with numpy arrays as projections
        '''
        # Number of non-negative elements in the projection
        non_neg = int(np.ceil(np.sqrt(dimension)))
        projections = []
        # We are going to compute k projections
        for i in range(self.k):
            # Select non_neg random indexes to make the projection
            ind = np.random.choice(dimension, replace=False, size=non_neg)
            # Initialize it to zeroes
            proj = np.zeros(dimension)
            # The non-negative elements are drawn from a normal distribution
            proj[ind]=np.random.normal(size=non_neg)
            projections.append(proj)
        return projections

    def getBin(self, hist_limits, value):
        '''
        @brief Function that given a value, it returns the bin it belongs to for the histogram
        @param self
        @param hist_limits Limits for each bin of the histogram
        @param value Value to check for the bin
        @return It returns the index corresponding to the bin
        '''
        bin=-1
        for i in range(len(hist_limits)):
            if value<hist_limits[i]:
                bin=i
                break
        return bin-1

    def runMethod(self):
        '''
        @brief This is the implementation of the LODA algorithm
        '''
        # We compute first all projections
        random_projections = self.getRandomProjections(len(self.dataset[0]))
        # Initialize the histograms and the projected data
        histograms = [[]]*self.k
        Z = [[]]*self.k
        # For each instance of the dataset
        for j in range(len(self.dataset)):
            # For each projection
            for i in range(self.k):
                # Compute the 1D projection
                Z[i].append(np.dot(self.dataset[j].T, random_projections[i]))
        # Compute the k histograms with the data
        for i in range(self.k):
            histograms[i]=np.histogram(Z[i], bins = self.n_bins)

        # Initialize the scores to zero
        self.outlier_score = np.array([0]*len(self.dataset))
        # For each instance
        for i in range(len(self.dataset)):
            prob = []
            # For each histogram
            for j in range(self.k):
                # Compute the projection
                z = np.dot(self.dataset[i].T, random_projections[j])
                # Check the bin for the projection
                bin = self.getBin(histograms[j][1], z)
                # Obtain the probability linked to z in the histogram
                prob.append(histograms[j][0][bin]/np.sum(histograms[j][0]))
            prob = np.array(prob)
            # Compute the score with the probabilities
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
