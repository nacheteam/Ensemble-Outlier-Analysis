#!# -*- coding: utf-8 -*-

import sys
sys.path.append('../models/')
import time

# Import all models
from trinity import TRINITY
from KernelMahalanobis import KernelMahalanobis
from hics import HICS
from loda import LODA
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

def getAccuracy(model, labels, my_model):
    '''
    @brief Function that computes the accuracy
    @param model Model to check for anomalies
    @param labels Ground truth of the dataset
    @param my_model If True it is one of our models, if False then is one of pyod
    @return It returns a number between 0 and 1 being 1 the highest rate of accuracy and
    0 the lowest.
    '''
    # Count the number of outliers in the dataset
    unique, counts = np.unique(labels, return_counts=True)
    noutliers = dict(zip(unique, counts))[1]
    # Obtain the same number of outliers based on the scores
    outliers = []
    if my_model:
        outliers = model.getOutliersBN(noutliers)
    else:
        scores = np.array(model.decision_scores_)
        outliers = scores.argsort()[-noutliers:][::-1]
    # Check the labels to see the accuracy
    correct = 0
    for out in outliers:
        if labels[out]==1:
            correct+=1
    return correct/noutliers

def writeResults(modelname, datasetname, model, accuracy, datasetnumber, time):
    '''
    @brief Function that writes the results of the execution of a model to a file
    @param modelname String with the name of the model
    @param datasetname Name of the dataset (string)
    @param model Object with the model fitted
    @param accuracy Float between 0 and 1 representing the accuracy
    @param datasetnumber Number of the dataset in the list of models
    @param time Time taken executing the model for the dataset
    '''
    f = open("./exp1/own/" + modelname + "_" + datasetname + "_" + str(datasetnumber) + ".txt", "w")
    f.write("Model: " + modelname + "\n")
    f.write("Dataset " + str(datasetnumber) + ": " + datasetname + "\n")
    f.write("Time taken: " + str(time) + " seg.\n")
    f.write("Accuracy: " + str(accuracy) + "\n")
    if accuracy!=None:
        f.write("@scores\n")
        for score in model.outlier_score:
            f.write(str(score) + "\n")
    f.close()

# This is based on executing the script from the folder experiments
ROUTE = "../datasets/outlier_ground_truth/"
# List of datasets
datasets = ["annthyroid.mat", "arrhythmia.mat", "breastw.mat", "cardio.mat", "glass.mat", "ionosphere.mat", "letter.mat", "lympho.mat", "mammography.mat", "mnist.mat", "musk.mat", "optdigits.mat", "pendigits.mat", "pima.mat", "satellite.mat", "satimage-2.mat", "speech.mat", "thyroid.mat", "vertebral.mat", "vowels.mat", "wbc.mat", "wine.mat"]
# List of models and names
models = [TRINITY(verbose=True), KernelMahalanobis(), OUTRES(verbose=True), LODA(), HICS(verbose=True)]
models = [TRINITY(verbose=True)]
names = ["TRINITY", "Mahalanobis Kernel", "OUTRES", "LODA", "HICS"]
names = ["TRINITY"]
accuracies = []

for name, model in zip(names, models):
    print("\n\n#################################################################")
    print("MODEL " + name + " " + str(names.index(name)+1) + "/" + str(len(names)))
    print("#################################################################")
    acc = []
    for dat in datasets:
        if name =="OUTRES" and dat in ["arrhythmia.mat", "mnist.mat", "musk.mat", "speech.mat", "cardio.mat", "ionosphere.mat", "letter.mat", "lympho.mat", "optdigits.mat", "satellite.mat", "satimage-2.mat", "wbc.mat"]:
            result = None
        elif name =="HICS" and dat in ["arrhythmia.mat", "mnist.mat", "musk.mat", "speech.mat", "cardio.mat", "ionosphere.mat", "letter.mat", "lympho.mat", "optdigits.mat", "satellite.mat", "satimage-2.mat", "wbc.mat", "pendigits.mat"]:
            result = None
        else:
            print("Computing dataset " + dat + " " + str(datasets.index(dat)+1) + "/" + str(len(datasets)))
            # Read dataset
            dataset, labels = readDataset(ROUTE + dat)
            print("The dataset has " + str(len(dataset)) + " number of instances with dimensionality " + str(len(dataset[0])))
            # Fit the model
            start = time.time()
            ker = fitModel(model, dataset)
            end = time.time()
            time_taken = end - start
            # Get accuracy
            result = getAccuracy(model, labels, True)
        acc.append(result)
        if result==None:
            print("Accuracy: None")
        else:
            print("Accuracy: " + str(result*100) + "%")
        # Write the results to a file
        writeResults(name, dat, model, result, datasets.index(dat)+1, time_taken)
    accuracies.append(acc)

print("\n\n\n")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
print("########################################################################")
print("RESULTS")
print("########################################################################\n\n")
for name, acc in zip(names, accuracies):
    print("\n\n#################################################################")
    print("MODEL " + name)
    print("#################################################################")
    for data, ac in zip(datasets, acc):
        if ac==None:
            print("Accuracy: None")
        else:
            print("The accuracy in the dataset " + data + " was " + str(ac*100) + "%")
