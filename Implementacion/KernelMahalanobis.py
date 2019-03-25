#!# -*- coding: utf-8 -*-
import numpy as np
import sklearn
from scipy import stats
from scipy.spatial import distance_matrix

# For plotting
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class KernelMahalanobis:
    '''
    Implements the Kernel Mahalanobis Ensemble method for Outlier Detection.
    The implementation is picked from Aggarwal, Charu C., Sathe, Saket Outlier Ensembles

    Parameters
    ------------
    iter_number: number of iterations for the subsampling loop.
    '''

    def __init__(self,iter_number,data,data_lenght):
        '''
        @brief Function that initialises the KernelMahalanobis class
        @param self
        @param iter_number Number of times subsampling is going to be applied
        @param data Dataset on which to apply the method
        @param data_lenght size of the dataset
        '''
        self.dataset=data
        self.niter=iter_number
        self.datasize=data_lenght
        self.calculations_done=False

    def runMethod(self):
        '''
        @brief Function that executes the Kernel Mahalanobis method. The results are
        stored on the variable self.scores
        @param self
        '''
        total_scores = None
        empty_scores = True
        # For each iteration
        for i in range(self.niter):
            print("Iteración "  + str(i+1) + "/" + str(self.niter))
            # Pull a random integer in (min{50,n},min{1000,n})
            s = np.random.randint(low=min([50,self.datasize]),high=min([1000,self.datasize]))
            # Create the subsample with size s
            subsample_indices = np.random.randint(self.datasize,size=s)
            subsample = self.dataset[subsample_indices,:]
            # Create the similarity matrix with the subsampled data (Usea the Euclidean distance, not this one https://stats.stackexchange.com/questions/78503/how-to-make-similarity-matrix-from-two-distributions)
            #sim_matrix = createSimilarityMatrix(subsample,subsample)
            sim_matrix = distance_matrix(subsample,subsample)
            # Use SVD to decompose S = QΔ^2Q^t
            Q,delta_sq,Qt = np.linalg.svd(sim_matrix)
            delta = np.diag(np.sqrt(delta_sq))
            Q = np.matrix(Q)
            delta_sq = np.matrix(delta_sq)
            # Obtain non-zero indices of Q vectors and obtain the correspondent Qk and Δk
            non_zero_ind = list(set(np.nonzero(Q)[1]))
            Qk=Q[:,non_zero_ind]
            deltak = delta[:,non_zero_ind]
            # Build the similarity matrix of the points out of the sample
            out_indices = list(set(range(self.datasize)).difference(set(subsample_indices)))
            out_subsample = self.dataset[out_indices,:]
            #out_sim_matrix = createSimilarityMatrix(self.dataset,subsample)
            out_sim_matrix = distance_matrix(self.dataset,subsample)
            # Build the total embedding and standardize
            #D = np.stack(np.dot(Qk,deltak),np.dot(out_sim_matrix,np.dot(Qk,np.linalg.inv(deltak))))
            D = np.dot(out_sim_matrix,np.dot(Qk, np.linalg.inv(deltak)))
            D_stand = stats.zscore(D)
            mean = D_stand.mean(0).reshape(-1)
            # Compute the score
            dimensions = len(D_stand[0])
            score=[]
            for j in range(len(D_stand)):
                score.append(np.linalg.norm(D_stand[j]-mean)/dimensions)
            if empty_scores:
                empty_scores=False
                total_scores=np.array(score)/self.niter
            else:
                total_scores=total_scores+np.array(score)/self.niter
        self.scores = np.array(total_scores)
        self.calculations_done=True

    def obtainResults(self):
        '''
        @brief Function that, given the calculations donde by runMethod and the
        data stored with it, gives back some statistical information about the results.
        @param feature_names Names of the characteristics or features involved in
        the dataset
        '''
        if not self.calculations_done:
            print("First you need to apply the method runMethod.")
            exit()

        print("\n\n\n##########################################################")
        print("Statistical information about the outliers")
        print("##########################################################\n\n")
        # First the mean, stdv and variance
        mean = np.mean(self.scores)
        stdv = np.std(self.scores)
        variance = np.var(self.scores)
        print("The mean of the scores is: " + str(mean))
        print("The standard deviation of the scores is: " + str(stdv))
        print("The variance of the scores is: " + str(variance))

        # We will say that the data whose score is more than 90% of the maximum
        # score are outliers, that is the percentile 90.
        border = 0.9*(max(list(self.scores))-min(list(self.scores)))+min(list(self.scores))
        outliers_index = np.where(self.scores>=border)
        # We store the outliers index
        self.outliers=list(outliers_index[0])
        print("The outliers are the elements with indexes: " + str(self.outliers))

        # Then we will plot the scores colouring the outliers in red.
        plt.scatter(list(range(len(self.scores))),self.scores,c="b",label="Normal data")
        plt.scatter(np.array(list(range(len(self.scores))))[outliers_index],self.scores[outliers_index],c="r",label="Outliers")
        plt.xlabel("Data")
        plt.ylabel("Score")
        plt.title("Scatter Plot of the scores, outliers in red.")
        plt.legend()
        plt.show()
