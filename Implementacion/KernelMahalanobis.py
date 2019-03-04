#!# -*- coding: utf-8 -*-
import numpy as np
import sklearn
from scipy import stats
import pdb

def createSimilarityMatrix(sample1,sample2):
    '''
    @brief Function that performs the similarity matrix between two samples of a dataset
    using the Euclidean norm
    @param sample1 The first sample to obtain the distances
    @param sample2 The second sample to obtain the distances
    @return Gives back a NumPy matrix object with the distance of the vector i-th and j-th
    in the position (i,j)
    '''
    sim_matrix = np.matrix(np.zeros(shape=(len(sample1),len(sample2))))
    for i in range(len(sample1)):
        for j in range(len(sample2)):
            sim_matrix[i,j]=np.linalg.norm(sample1[i]-sample2[j])
    return sim_matrix

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
        @brief Función que inicializa la clase KernelMahalanobis
        @param self Objeto con el que se llama
        @param iter_number Número de iteraciones para realizar subsampling
        @param data Conjunto de datos sobre el que se quiere saber los scores
        que cuantifican cómo de anómalos son los datos
        @param data_lenght Tamaño del conjunto de datos
        '''
        self.dataset=data
        self.niter=iter_number
        self.datasize=data_lenght

    def runMethod(self):
        '''
        @brief Función que ejecuta el método Kernel Mahalanobis
        @param self Objeto con el que se llama
        @return Devuelve una lista de scores que cuantifican cómo de anómalos son
        los puntos
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
            sim_matrix = createSimilarityMatrix(subsample,subsample)
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
            out_sim_matrix = createSimilarityMatrix(self.dataset,subsample)
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
        return total_scores
