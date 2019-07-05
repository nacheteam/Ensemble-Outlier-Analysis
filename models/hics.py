from base import EnsembleTemplate
import numpy as np

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
