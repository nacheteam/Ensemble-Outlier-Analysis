from base import EnsembleTemplate
import numpy as np
from itertools import combinations
from pyod.models.lof import LOF
from pyod.models.cof import COF
from pyod.models.cblof import CBLOF
from pyod.models.loci import LOCI
from pyod.models.hbos import HBOS
from pyod.models.sod import SOD

from multiprocessing import Pool
from functools import reduce

OUTLIER_RANKING_POS = ["lof", "cof", "cblof", "loci", "hbos", "sod"]

class HICS(EnsembleTemplate):
    """
    Implementation of the algorithm HiCS: High Contrast Subspaces for Density-Based
    Outlier Ranking. Authors: Fabian Keller, Emmanuel Müller, Klemens Böhm

    Parameters
    ------------
    * outlier_rank: method used to evaluate the instances as outliers, it should be a
    density-based method. The original proposal by the authors was LOF, used by the
    algorithm by default. Other choices are available: COF, CBLOF, LOCI, HBOS or SOD.

    * M: number of Monte Carlo iterations to perform in the subspace contrast calculations,
    by default 100.

    * alpha: the desired average size of the test statistic. This will be calculated
    dynamically by using this parameter.

    * numCandidates: Maximum number of candidates in each dimension to be a real high contrast subspace

    * maxOutputSpaces: Maximum number of subspaces to have at the end
    """

    def __init__(self, outlier_rank="lof", contamination=0.1, M=100, alpha=0.1, numCandidates=500, maxOutputSpaces=1000, numThreads=8, verbose=False):
        '''
        @brief Constructor of the class
        @param self
        @param outlier_rank Method to score each instance of the dataset in each high contrast subspace
        @param contamination Percentage of outliers expected to be in the dataset
        @param M Number of Monte Carlo iterations to perform in the subspace contrast calculations,
        by default 100. Default is 100
        @param alpha The desired average size of the test statistic. This will be calculated
        dynamically by using this parameter. Default is 0.1
        @param numCandidates Maximum number of candidates in each dimension to be a real high contrast subspace.
        Default is 500
        @param maxOutputSpaces Maximum number of subspaces to have at the end. Default is 1000.
        '''
        assert outlier_rank in OUTLIER_RANKING_POS, ("The method \"" + outlier_rank + "\" is not available, try " + str(OUTLIER_RANKING_POS))
        self.outlier_rank = outlier_rank
        self.contamination = contamination
        self.M = M
        self.alpha = 0.1
        self.numCandidates=numCandidates
        self.maxOutputSpaces = maxOutputSpaces
        self.calculations_done = False
        self.numThreads = numThreads
        self.verbose=verbose

    def computeContrast(self, subspace):
        '''
        @brief Function that computes the contrast for a given subspace
        @param subspace Numpy array with the indexes of the features that define the subspace
        @return It returns a float representing the contrast of the subspace
        '''
        # We set the adaptative size of the test
        size = int(len(self.dataset)*np.power(self.alpha, len(subspace)))
        # Number of instances in the dataset
        N = len(self.dataset)
        deviation = 0
        # We repeat the process M times
        for i in range(1,self.M+1):
            # Shuffle the features
            #np.random.shuffle(subspace)
            # This is the comparison attribute for the test, so it will stay untouched
            comparison_attr=np.random.randint(low=0, high=len(subspace))
            # List of booleans that masks the instances of the dataset selected
            selected_objects = np.array([True]*N)
            # Select random indexes
            selected_objects[reduce(np.union1d,np.array([np.random.choice(N,size=size,replace=False) for _ in range(len(subspace)-1)]))]=False
            # With the sample given by the mask selected_objects we compute the deviation
            deviation+=self.computeDeviation(subspace[comparison_attr], selected_objects, subspace)
        # Finally the contrast is the average of all deviations
        return deviation/self.M

    def computeDeviation(self, comparison_attr, selected_objects, subspace):
        '''
        @brief Function that computes the deviation of the marginal distribution
        given a fixed attribute, a sample and the condition given as a subspace
        @param comparison_attr This is the comparison attribute
        @param selected_objects Mask that sets a sample of the dataset
        @param subspace Subspace or condition to calculate the deviation
        @return It returns a float giving the deviation
        '''
        max = 0
        # For each instance of the dataset
        for d in self.dataset:
            # This is the cumulative value for all elements in the dataset
            cumul1 = np.sum(self.dataset[:,comparison_attr][self.dataset[:,comparison_attr]<d[comparison_attr]])
            # This is the cumulative value for the selected_objects aka the sample
            sel = self.dataset[:,comparison_attr][selected_objects]
            cumul2 = np.sum(sel[sel<d[comparison_attr]])
            # Finally we compute the average in both cases
            fa = cumul1/len(self.dataset)
            fb = cumul2/len(self.dataset)
            # The difference in absolute value is the deviation
            subs = np.absolute(fa-fb)
            # We return the biggest of the deviations obtained
            if subs>max:
                max = subs
        return max

    def hicsFramework(self):
        '''
        @brief This function computes the high contrast subspaces on which to score the outlier
        @return It returns a numpy array containing the high contrast subspaces
        '''
        # Ordered subspaces by dimension type: list of numpy arrays of numpy arrays
        # this means that in each positions there would be the subspaces of the corresponding
        # dimension in the form of a list of subspaces which are numpy arrays
        all_subspaces = []
        # Record of the contrast for each subspace in each dimension, same shape as all_subspaces
        all_contrasts = []
        # For all dimensions starting from dimension 2 (correlation has no sense on dimension 1)
        for dimension in range(2,len(self.dataset[0])):
            if self.verbose:
                print("Calculating subspaces in dimension " + str(dimension) + "/" + str(len(self.dataset[0])))
            candidates = []
            contrasts = []
            # This list will keep the indexes of the redundant subspaces, those are d-dimensional subspaces with d+1-dimensional
            # subspaces containing them with higher contrast
            redundant = []
            # For dimension 2 we just obtain all possible indexes and make all combinations
            if dimension==2:
                # Calculate the candidates as all possible combinations
                indexes = list(range(len(self.dataset[0])))
                candidates = np.array([np.array(list(comb)) for comb in list(combinations(indexes,dimension))])
                # Compute the contrasts
                cont = 0
                p = Pool(self.numThreads)
                while cont+self.numThreads<len(candidates):
                    contrasts = contrasts + p.map(self.computeContrast,candidates[cont:cont+self.numThreads])
                    cont+=self.numThreads
                    print("Computed " + str(cont) + "/" + str(len(candidates)))
                p = Pool(len(candidates)-cont)
                contrasts = contrasts + p.map(self.computeContrast,candidates[cont:])
                print("Computed " + str(len(candidates)) + "/" + str(len(candidates)))
            else:
                # We need to calculate now the indexes starting from a previous subspace
                # We record the parent of each subspace to check for redundancy
                parents = []
                # For all subspaces with one dimension less
                for i in range(len(all_subspaces[-1])):
                    # We only consider new indexes, those are the ones not uses in the father subspace
                    indexes = list(set(list(range(len(self.dataset[0])))).difference(set(all_subspaces[-1][i])))
                    # For each new index
                    for ind in indexes:
                        # We calculate the new candidate as the same subspace appending the index
                        new_can = np.append(all_subspaces[-1][i],ind)
                        # Now we check that the candidate wasn't in the list before
                        new = True
                        for previous in candidates:
                            if len(np.intersect1d(previous,new_can))==len(new_can):
                                new = False
                        if new:
                            candidates.append(new_can)
                            parents.append(i)
                # Compute the contrasts
                cont = 0
                p = Pool(self.numThreads)
                while cont+self.numThreads<len(candidates):
                    contrasts = contrasts + p.map(self.computeContrast,candidates[cont:cont+self.numThreads])
                    cont+=self.numThreads
                    print("Computed " + str(cont) + "/" + str(len(candidates)))
                p = Pool(len(candidates)-cont)
                contrasts = contrasts + p.map(self.computeContrast,candidates[cont:])
                print("Computed " + str(len(candidates)) + "/" + str(len(candidates)))

                # Check for redundancy
                for i in range(len(parents)):
                    if contrasts[i]>all_contrasts[-1][parents[i]]:
                        redundant.append(parents[i])

            candidates = np.array(candidates)
            contrasts = np.array(contrasts)
            # If there are redundant subspaces
            if redundant!=[]:
                if self.verbose:
                    print("Now deleting redundant subspaces in dimension " + str(dimension) + ", " + str(len(redundant)) + " subspaces removed.")
                # Delete those ones
                non_redundant_sub = np.delete(all_subspaces[-1], redundant)
                # Update the subspaces
                all_subspaces[-1]=non_redundant_sub
            # Sort from higher contrast to lower and only get numCandidates number of subspaces if available
            if len(candidates)>self.numCandidates:
                all_subspaces.append(candidates[contrasts.argsort()[-self.numCandidates:][::-1]])
                all_contrasts.append(contrasts[contrasts.argsort()[-self.numCandidates:][::-1]])
            else:
                all_subspaces.append(candidates)
                all_contrasts.append(contrasts)
        # We flatten the numpy array to obtain only a list of subspaces and contrasts
        subspaces = np.array(all_subspaces).flatten()
        contrasts = np.array(all_contrasts).flatten()
        # We only give the maxOutputSpaces with higher contrast if available
        if len(subspaces)>self.maxOutputSpaces:
            return subspaces[contrasts.argsort()[-self.maxOutputSpaces:][::-1]]
        return subspaces

    def runMethod(self):
        '''
        @brief This function is the actual implementation of HICS
        '''
        if self.verbose:
            print("Calculating the subspaces\n")
        # First we obtain the high contrast subspaces
        subspaces = self.hicsFramework()

        if self.verbose:
            print("Now calculating the scoring\n")
        # We initialize the scores for each instance as 0
        scores = np.zeros(len(self.dataset))
        # For each subspace
        for sub in subspaces:
            # We place the corresponding scorer according to parameter
            scorer = None
            if self.outlier_rank=="lof":
                scorer = LOF()
            elif self.outlier_rank=="cof":
                scorer = COF()
            elif self.outlier_rank=="cblof":
                scorer = CBLOF()
            elif self.outlier_rank=="loci":
                scorer = LOCI()
            elif self.outlier_rank=="hbos":
                scorer = HBOS()
            elif self.outlier_rank=="sod":
                scorer = SOD()
            # Fits the scorer with the dataset
            scorer.fit(self.dataset)
            # Adds the scores obtained to the global ones
            scores = scores+scorer.decision_scores_
        # Compute the average
        self.outlier_score = scores/len(subspaces)
        # Marks the calculations as done
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
