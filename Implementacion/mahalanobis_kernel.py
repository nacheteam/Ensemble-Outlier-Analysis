#!# -*- coding: utf-8 -*-
import numpy as np
import sklearn

class KernelMahalanobis:
    '''
    Implements the Kernel Mahalanobis Ensemble method for Outlier Detection.
    The implementation is picked from Aggarwal, Charu C., Sathe, Saket Outlier Ensembles

    Parameters
    ------------
    iter_number: number of iterations for the subsampling loop.
    '''

    def __init__(self,iter_number,data,data_lenght):
        self.dataset=data
        self.niter=iter_number
        self.datasize=data_lenght

    def runMethod(self):
        # For each iteration
        for i in range(self.niter):
            # Pull a random integer in (min{50,n},min{1000,n})
            s = np.random.randint(low=min([50,self.datasize]),high=min([1000,self.datasize]))
            # Create the subsample with size s
            subsample_indices = np.random.randint(self.datasize,size=s)
            subsample = self.dataset[subsample_indices]
            # Create the similarity matrix with the subsampled data
            sim_matrix = np.dot(subsample,np.transpose(subsample))
            # Use SVD to decompose S = QΔ^2Q^t
            Q,delta_sq,Qt = np.linalg.svd(sim_matrix)
            delta = np.diag(np.sqrt(delta_sq))
            # Obtain non-zero indices of Q vectors and obtain the correspondent Qk and Δk
            non_zero_ind = np.nonzero(Q)[0]
            Qk=Q[non_zero_ind]
            deltak = delta[non_zero_ind]
            # Build the similarity matrix of the points out of the sample
            out_indices = set(range(100)).difference(set(subsample_indices))
            out_subsample = self.dataset[out_indices]
            out_sim_matrix = np.dot(out_subsample,np.transpose(out_subsample))
            
