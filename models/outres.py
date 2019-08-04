from base import EnsembleTemplate
import numpy as np

from scipy.stats import kstest
from scipy.special import gamma

import multiprocessing

class OUTRES(EnsembleTemplate):
    """
    Implementation of the algorithm OUTRES: Statistical Selection of Relevant Subspace
    Projections for Outlier Ranking. Authors: Emmanuel Müller, Matthias Schiffer and Thomas Seidl.

    Parameters
    ------------
    * alpha: level of confidence used in the Kolmogorov Smirnov test. This is not a
    real parameter given by the paper, the default value (as said in the paper) is 0.01
    but we give the possibility to change it.
    """

    def __init__(self, contamination=0.1, alpha=0.01, numThreads=8, verbose=False):
        '''
        @brief Constructor of the class
        @param self
        @param contamination Percentage of outliers expected to be in the dataset
        @param alpha Parameter for the Kolmogorov Smirnov test to check if the
        data is uniformly distributed
        '''
        self.alpha = alpha
        self.contamination=contamination
        self.verbose = verbose
        self.numThreads=numThreads

    def isRelevantSubspace(self, subspace, neighborhood):
        for sub in self.checked_subspaces:
            if len(np.intersect1d(sub, subspace))==len(subspace):
                return False

        projection = self.dataset[:,subspace][neighborhood]
        for i in range(len(subspace)):
            min = np.amin(projection[i])
            max = np.amax(projection[i])
            d,p = kstest(projection[i], "uniform", args=(min, max-min))
            if p<=self.alpha:
                return False
        return True

    def computeHOptimal(self, d):
        f1 = (8*gamma(d/2 + 1))/(np.power(np.pi, d/2))
        f2 = d+4
        f3 = np.power(2*np.sqrt(np.pi),d)
        n = len(self.dataset)
        f4 = np.power(n, -1/(d+4))
        return f1*f2*f3*f4

    def computeEpsilon(self, subspace):
        return 0.5*(self.computeHOptimal(len(subspace))/self.computeHOptimal(2))

    def computeNeighborhood(self, subspace, instance):
        projection = self.dataset[:,subspace]
        tile = np.tile(projection[instance], len(self.dataset)).reshape((len(self.dataset),len(projection[instance])))
        distances = np.linalg.norm(projection-tile, axis=1)
        neighborhood = np.where(distances<self.epsilons[len(subspace)])[0]
        return neighborhood[neighborhood!=instance]

    def computeKernel(self, x):
        return 1-np.power(x,2)

    def computeDensity(self, subspace, neighborhood, instance):
        projection = self.dataset[:,subspace]
        tile = np.tile(projection[instance], len(self.dataset)).reshape((len(self.dataset),len(projection[instance])))
        return np.sum(self.computeKernel(np.linalg.norm(projection-tile, axis=1))/self.computeEpsilon(subspace))/len(self.dataset)

    def computeDeviation(self, subspace, neighborhood, instance, density):
        densities = np.array([])
        for neig in neighborhood:
            local_neigborhood = self.computeNeighborhood(subspace, neig)
            densities = np.append(densities,self.computeDensity(subspace, local_neigborhood, neig))
        mean = np.mean(densities)
        stdv = np.std(densities)
        return (mean-density)/(2*stdv)

    # First of all, instance is the index of the actual instance in the dataset
    def outres(self, instance, subspace):
        available_indexes = list(set(list(range(len(self.dataset[0])))).difference(set(list(subspace))))
        for index in available_indexes:
            new_subspace = np.append(subspace, int(index)).astype(int)
            neighborhood = self.computeNeighborhood(new_subspace, instance)
            if self.isRelevantSubspace(new_subspace, neighborhood):
                density = self.computeDensity(new_subspace, neighborhood, instance)
                deviation = self.computeDeviation(new_subspace, neighborhood, instance, density)
                if deviation>=1:
                    if self.verbose:
                        print("This instance is outlying in the subspace " + str(new_subspace))
                    # The scores are equal to 1 at first and 1 means no outlierness and 0 means very outlying
                    self.outlier_score[instance]*=density/deviation
                self.outres(instance, new_subspace)
            self.checked_subspaces.append(new_subspace)


    def runMethod(self):
        '''
        @brief This function is the actual implementation of OUTRES
        '''
        self.epsilons = [self.computeEpsilon(list(range(n))) for n in range(len(self.dataset[0]))]

        self.outlier_score = np.ones(len(self.dataset))
        for i in range(1):
            self.checked_subspaces = []
            if self.verbose:
                print("Computing the instance " + str(i+1) + "/" + str(len(self.dataset)))
            for j in range(len(self.dataset[0])):
                self.outres(i,np.array([j]))
        self.outlier_score = np.ones(len(self.dataset))-self.outlier_score
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
