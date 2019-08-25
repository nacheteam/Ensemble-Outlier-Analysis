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

    def __init__(self, contamination=0.1, alpha=0.01, verbose=False, experiment=False):
        '''
        @brief Constructor of the class
        @param self
        @param contamination Percentage of outliers expected to be in the dataset
        @param alpha Parameter for the Kolmogorov Smirnov test to check if the
        data is uniformly distributed
        @param verbose Makes the algorithm to print some information during execution
        @param experiment Indicates if the experiment is working. Nothing to do with the algorithm
        '''
        self.alpha = alpha
        self.contamination=contamination
        self.verbose = verbose
        self.experiment = experiment

    def isRelevantSubspace(self, subspace, neighborhood):
        '''
        @brief Function that tells if a subspace is relevant, this is that the projection
        of the dataset over the subspace is not distributed uniformly in the neighborhood
        @param self
        @param subspace Subspace to check
        @param neighborhood neighborhood in which to check the projection
        @return It returns True if the subspace is relevant, False in other case.
        '''
        # We check first if we have already considered this subspace. If so, it is not relevant anymore.
        for sub in self.checked_subspaces:
            if len(np.intersect1d(sub, subspace))==len(subspace):
                return False

        # Make the projection
        projection = self.dataset[:,subspace][neighborhood]
        if len(projection)==0:
            return False
        # We check for each subspace if the 1-dimensional data is uniformly distributed
        for i in range(len(subspace)):
            min = np.amin(projection[:,i].reshape(-1))
            max = np.amax(projection[:,i].reshape(-1))
            # We do it using the Kolmogorov-Smirnov test
            d,p = kstest(projection[:,i], "uniform", args=(min, max-min))
            # If the null hypothesis is not rejected, this means the data follow a uniform distribution
            if p<=self.alpha:
                return False
        return True

    def computeHOptimal(self, d):
        '''
        @brief Function that calculates the Hoptimal
        @param self
        @param d Parameter, usually the dimensionality of the subspace
        @return It returns a numerical value.
        '''
        f1 = (8*gamma(d/2 + 1))/(np.power(np.pi, d/2))
        f2 = d+4
        f3 = np.power(2*np.sqrt(np.pi),d)
        n = len(self.dataset)
        f4 = np.power(n, -1/(d+4))
        return f1*f2*f3*f4

    def computeEpsilon(self, subspace):
        '''
        @brief Function to compute the epsilon of the adapatative neighborhood
        @param self
        @params subspace Subspace considered to compute the epsilon
        @return It returns a numerical value
        '''
        return 0.5*(self.computeHOptimal(len(subspace))/self.computeHOptimal(2))

    def computeNeighborhood(self, subspace, instance):
        '''
        @brief This function computes the adaptative neighborhood
        @param subspace Subspace in which to compute the neighborhood
        @param instance Instance considered as the centroid of the neighborhood (index of the element)
        @return It returns a numpy array containing the indexes of the neighborhood
        '''
        # First we compute the projection
        projection = self.dataset[:,subspace]
        # We compute a numpy array of the distances of all the elements to the instance
        tile = np.tile(projection[instance], len(self.dataset)).reshape((len(self.dataset),len(projection[instance])))
        distances = np.linalg.norm(projection-tile, axis=1)
        # We keep only the ones that are close enough (epsilon distance as max)
        neighborhood = np.where(distances<self.epsilons[len(subspace)])[0]
        # We exclude the instance itself
        return neighborhood[neighborhood!=instance]

    def computeKernel(self, x):
        '''
        @brief Function that computes the Epanechnikov kernel with scalar factor 1
        @param self
        @param x Number between 0 and 1.
        @return It returns a numerical value
        '''
        return 1-np.power(x,2)

    def computeDensity(self, subspace, neighborhood, instance):
        '''
        @brief This is the function that computes the density
        @param self
        @param subspace Subspace in which to compute the density
        @param neighborhood Adaptative neighborhood for the instance in the subspace
        @param instance Index of the instance considered at the moment
        @return It return a numerical value.
        '''
        # Compute the projection
        projection = self.dataset[:,subspace]
        # Compute the density
        tile = np.tile(projection[instance], len(self.dataset)).reshape((len(self.dataset),len(projection[instance])))
        return np.sum(self.computeKernel(np.linalg.norm(projection-tile, axis=1))/self.computeEpsilon(subspace))/len(self.dataset)

    def computeDeviation(self, subspace, neighborhood, instance, density):
        '''
        @brief Function that computes the deviation
        @param self
        @param subspace Subspace considered to compute the deviation
        @param neighborhood Adaptative neighborhood for the instance in the subspace
        @param instance Instance to compute the deviation
        @param density Density value of the instance
        @return It returns a numerical value
        '''
        # First we need to compute the density for all the neighbors
        densities = np.array([])
        for neig in neighborhood:
            local_neigborhood = self.computeNeighborhood(subspace, neig)
            densities = np.append(densities,self.computeDensity(subspace, local_neigborhood, neig))
        # We compute the mean and the standard deviation
        mean = np.mean(densities)
        stdv = np.std(densities)
        # Return the deviation
        return (mean-density)/(2*stdv)

    def outres(self, instance, subspace):
        '''
        @brief Main loop of the outres algorithm
        @param self
        @param instance Instance to compute the outres score
        @param subspace Initial subspace of dimension 1
        '''
        # First we compute the indexes of the features that are not used in the actual subspace
        available_indexes = list(set(list(range(len(self.dataset[0])))).difference(set(list(subspace))))
        # For each available index we are going to check
        for index in available_indexes:
            # We make the new subspace adding the index
            new_subspace = np.append(subspace, int(index)).astype(int)
            # We compute the adaptative neighborhood
            neighborhood = self.computeNeighborhood(new_subspace, instance)
            # If the subspace is relevant
            if self.isRelevantSubspace(new_subspace, neighborhood):
                # Compute the density and deviation
                density = self.computeDensity(new_subspace, neighborhood, instance)
                deviation = self.computeDeviation(new_subspace, neighborhood, instance, density)
                # If it is a high deviating instance in the subspace then we update the score
                if deviation>=1:
                    if self.verbose:
                        print("The instance " + str(instance+1) + " is outlying in the subspace " + str(new_subspace))
                    if self.experiment:
                        self.outliying_subspaces.append([instance, new_subspace, neighborhood])
                    # The scores are equal to 1 at first and 1 means no outlierness and 0 means very outlying
                    self.outlier_score[instance]*=density/deviation
                # We keep the process if the subspace was relevant
                self.outres(instance, new_subspace)
            # We add the subspace to the considered ones
            self.checked_subspaces.append(new_subspace)


    def runMethod(self):
        '''
        @brief This function is the actual implementation of OUTRES
        '''
        self.outliying_subspaces = []

        # First we compute all epsilons so we dont need to make this calculation more than once
        self.epsilons = [self.computeEpsilon(list(range(n))) for n in range(len(self.dataset[0])+1)]

        # We initialize the scores to one
        self.outlier_score = np.ones(len(self.dataset))
        # For each instance we run outres
        for i in range(len(self.dataset)):
            # Erase checked_subspaces
            self.checked_subspaces = []
            if self.verbose and i%25==0:
                print("Computing the instance " + str(i+1) + "/" + str(len(self.dataset)))
            # We run for each instance each index
            for j in range(len(self.dataset[0])):
                self.outres(i,np.array([j]))
        # At the end, score 1 means no outlierness and 0 100% outlier. We make 1-score
        # so we can keep the ascending order and now this will mean that 0 is no outlierness
        # and 1 is very outlying.
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
