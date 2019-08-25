#!# -*- coding: utf-8 -*-

import sys
sys.path.append('../models/')
import time

from outres import OUTRES

import numpy as np
# Set the seed for the experiment
np.random.seed(123456789)

# For reading the .mat files
import scipy.io

def checkForNan(d):
    '''
    @brief Function that checks for nans in the vector d
    @param d numpy array or list to check for nans
    @return It returs True if d contains a nan value, False in other case
    '''
    for e in d:
        if np.isnan(e):
            return True
    return False

def readDataset(name):
    '''
    @brief Function to read the dataset
    @paramm name Route to the dataset
    @return It returns two numpy arrays, first the dataset and second the labels
    '''
    # Load the file
    file = scipy.io.loadmat(name)
    data = []
    labels = []
    # Takes the data and labels only if there are no missing values
    for i in range(len(file['X'])):
        if not checkForNan(file['X'][i]):
            data.append(file['X'][i])
            labels.append(file['y'][i])
    # Parse them to numpy array
    data = np.array(data)
    labels = np.array(labels)
    del file
    return data, labels

def fitModel(model, dataset):
    '''
    @brief Function that fits the model
    @param model Model to fit
    @param dataset Dataset to fit the model
    @return It returns the model already fitted
    '''
    # Pyod and my own models work under fit
    model.fit(dataset)
    return model

def writeResults(datasetname, model, datasetnumber):
    '''
    @brief Function that writes the results of the execution of a model to a file
    @param datasetname Name of the dataset (string)
    @param model Object with the model fitted
    @param datasetnumber Number of the dataset in the list of models
    '''
    f = open("./exp2/outres_" + datasetname + "_" + str(datasetnumber) + ".txt", "w")
    f.write("Dataset " + str(datasetnumber) + ": " + datasetname + "\n")
    f.write("@outliying_subspaces\n")
    f.write("outlier ; subspace ; neighborhood\n")
    for outlier,subspace,neighborhood in model.outliying_subspaces:
        f.write(str(outlier) + ";")
        for s in subspace[:-1]:
            f.write(str(s) + ",")
        f.write(str(subspace[-1]))
        for neig in neighborhood[:-1]:
            f.write(str(neig) + ",")
        f.write(str(neighborhood[-1]))
        f.write("\n")
    f.close()

# This is based on executing the script from the folder experiments
ROUTE = "../datasets/outlier_ground_truth/"
# List of datasets
datasets = ["annthyroid.mat", "breastw.mat", "glass.mat", "mammography.mat", "pendigits.mat", "pima.mat", "thyroid.mat", "vertebral.mat", "vowels.mat", "wine.mat"]

print("\n\n#################################################################")
print("OBTAINING SUBSPACES AND OUTLIERS WITH OUTRES")
print("#################################################################")
for dat in datasets:
    outres = OUTRES(verbose=True, experiment=True)
    print("Computing dataset " + dat + " " + str(datasets.index(dat)+1) + "/" + str(len(datasets)))
    # Read dataset
    dataset, labels = readDataset(ROUTE + dat)
    print("The dataset has " + str(len(dataset)) + " number of instances with dimensionality " + str(len(dataset[0])))
    # Fit the model
    ker = fitModel(outres, dataset)
    # Write the results to a file
    writeResults(dat, outres, datasets.index(dat)+1)
