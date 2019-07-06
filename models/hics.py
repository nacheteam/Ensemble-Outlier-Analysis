from base import EnsembleTemplate
import numpy as np
from itertools import combinations

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

    * numCandidates:
    """

    def __init__(self, outlier_rank="lof", contamination=0.1, M=100, alpha=0.1, numCandidates=500, maxOutputSpaces=1000):
        assert outlier_rank in OUTLIER_RANKING_POS, ("The method \"" + outlier_rank + "\" is not available, try " + str(OUTLIER_RANKING_POS))
        self.outlier_rank = outlier_rank
        self.contamination = contamination
        self.M = M
        self.alpha = 0.1
        self.numCandidates=numCandidates
        self.maxOutputSpaces = maxOutputSpaces

    # A subspace is a subset of the indexes of the atributes
    def computeContrast(self, subspace):
        size = int(len(self.dataset)*np.power(self.alpha, len(subspace)))
        N = len(self.dataset)
        deviation = 0
        for i in range(1,self.M+1):
            np.random.shuffle(subspace)
            comparison_attr=np.random.randint(low=0, high=len(subspace))
            selected_objects = [True]*N
            for j in range(1,len(subspace)+1):
                if j!=comparison_attr:
                    random_block = np.random.choice(N,size=size, replace=False)
                    selected_objects[random_block] = False
            deviation+=self.deviation(subspace[comparison_attr], selected_objects, subspace)
        return deviation/self.M

    def computeDeviation(self, comparison_attr, selected_objects, subspace):
        max = 0
        for d in self.dataset:
            cumul1 = 0
            cumul2 = 0
            for i in range(len(self.dataset)):
                if self.dataset[i][comparison_attr]<d[comparison_attr]:
                    cumul1+=self.dataset[i][comparison_attr]
                    if selected_objects[i]:
                        cumul2+=self.dataset[i][comparison_attr]
            fa = cumul1/len(self.dataset)
            fb = cumul2/len(self.dataset)
            subs = np.absolute(fa-fb)
            if subs>max:
                max = subs
        return max

    def generateSubspaces(self):
        # Ordered subspaces by dimension type: list of numpy arrays of numpy arrays
        # this means that in each positions there would be the subspaces of the corresponding
        # dimension in the form of a list of subspaces which are numpy arrays
        all_subspaces = []
        all_contrasts = []
        for dimension in range(2,len(self.dataset[0])):
            candidates = np.array([])
            contrasts = np.array([])
            redundant = []
            # For dimension 2 we just obtain all possible indexes and make all combinations
            if dimension==2:
                # Calculate the candidates
                indexes = list(range(len(self.dataset[0])))
                candidates = np.array([np.array(list(comb)) for comb in list(combinations(indexes,dimension))])
                # Compute the contrasts
                contrasts = np.array([self.computeContrast(can) for can in candidates])

            else:
                # We need to calculate now the indexes starting from a previous subspace
                for i in range(len(all_subspaces[-1])):
                    indexes = list(set(list(range(len(self.dataset[0])))).difference(set(all_subspaces[-1][i])))
                    parent_should_be_removed = False
                    for ind in indexes:
                        candidates = np.append(candidates,np.append(all_subspaces[-1][i],ind))
                        contrasts = np.append(contrasts, self.computeContrast(candidates[-1]))
                        if contrasts[-1]>all_contrasts[-1][i]:
                            parent_should_be_removed=True
                    if parent_should_be_removed:
                        redundant.append(i)
            non_redundant_sub = np.delete(all_subspaces[-1], redundant)
            all_subspaces[-1]=non_redundant_sub
            # Sort from higher contrast to lower and only get numCandidates number of subspaces if available
            if len(candidates)>self.numCandidates:
                all_subspaces.append(candidates[contrasts.argsort()[-self.numCandidates:][::-1]])
                all_contrasts.append(contrasts[contrasts.argsort()[-self.numCandidates:][::-1]])
            else
                all_subspaces.append(candidates)
                all_contrasts.append(contrasts)
        subspaces = np.array(all_subspaces).flatten()
        contrasts = np.array(all_contrasts).flatten()
        return subspaces[contrasts.argsort()[-self.maxOutputSpaces:][::-1]]
