from base import EnsembleTemplate
import numpy as np

from scipy.stats import kstest
from scipy.special import gamma

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

    def __init__(self, contamination=0.1, alpha=0.01):
        '''
        @brief Constructor of the class
        @param self
        @param contamination Percentage of outliers expected to be in the dataset
        @param alpha Parameter for the Kolmogorov Smirnov test to check if the
        data is uniformly distributed
        '''
        self.alpha = alpha

    def isRelevantSubspace(self,subspace):
        '''
        @brief Function that, given a subspace it returns True if the subspace is
        relevant (the data is not following a uniform distribution) and False if the
        data is uniformly distributed in the subspace.
        @param subspace Numpy array with the indexes of the features chosen as
        the subspace
        @return It returns True or False depending wether the subspace is relevant
        or not
        '''
        d, pvalue = kstest(self.dataset[:,subspace], "uniform")
        return pvalue>self.alpha

    def hoptimal(self,dimensionality):
        return (8*gamma(dimensionality/2 + 1)/np.pow(np.pi, dimensionality/2))*(dimensionality+4)*(np.pow(2*np.sqrt(np.pi), dimensionality))*(np.pow(len(self.dataset), -1/(dimensionality+4)))

    def epsilon(self,subspace):
        # This could be changed, as self.hoptimal(2) remains constant during exec it could be set as a global constant maybe(?)
        return 0.5*(self.hoptimal(len(subspace))/self.hoptimal(2))

    # Hey buddy, instance is the INDEX of the element in the dataset
    def adaptativeNeighborhood(self,subspace, instance):
        neigbors_ind = []
        epsilon = self.epsilon(subspace)
        for i in range(len(self.dataset)):
            if i!=instance and np.linalg.norm(self.dataset[:,subspace][i]-self.dataset[:,subspace][instance])<=epsilon:
                neigbors_ind.append(i)
        return np.array(neigbors_ind)

    def kernel(self, x):
        return 1-np.pow(x,2)

    def density(self, subspace, instance):
        neig = self.adaptativeNeighborhood(subspace, instance)
        epsilon = self.epsilon(subspace)
        density = np.sum(kernel(np.linalg.norm(self.dataset[:,subspace]-np.array([self.dataset[:,subspace][instance]]*len(self.dataset)))/epsilon))
        return density/len(self.dataset)

    def deviation(self, subspace, mu, sigma, density):
        return (mu-density)/(2*sigma)

    def outres(self, instance, subspace):
        remaining_indexes = set(list(range(len(self.dataset[0])))).difference(set(list(subspace)))
        for ind in remaining_indexes:
            new_subs = np.append(subspace,ind)
            if self.isRelevantSubspace(new_subs):
                density = self.density(new_subs, instance)
                dens_neig = np.array([self.density(new_subs, ne) for ne in self.adaptativeNeighborhood(new_subs, instance)])
                mean = np.mean(dens_neig)
                stdv = np.std(dens_neig)
                deviation = self.deviation(new_subs, mean, stdv, density)
                if deviation>=1:
                    self.outlier_score[instance]*=density/deviation
                self.outres(instance,new_subs)

    def runMethod(self):
        '''
        @brief This function is the actual implementation of HICS
        '''
        for i in range(len(self.dataset)):
            outres(i,np.array([]))

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
