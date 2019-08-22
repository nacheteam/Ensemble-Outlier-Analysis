#!# -*- coding: utf-8 -*-

import sys
sys.path.append('../models/')
from trinity import TRINITY
from KernelMahalanobis import KernelMahalanobis
from hics import HICS
from loda import LODA
from outres import OUTRES

import numpy as np
np.random.seed(123456789)

# For reading the .mat files
import scipy.io
import h5py

version37 = ["http.mat"]

def checkForNan(d):
    for e in d:
        if np.isnan(e):
            return True
    return False

def readDataset(name, version37=False):
    '''
    @brief Function to read the dataset
    @paramm name Route to the dataset
    @return It returns two numpy arrays, first the dataset and second the labels
    '''
    # Load the file
    file=None
    if not version37:
        file = scipy.io.loadmat(name)
    else:
        f
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

def writeResults(modelname, datasetname, model, accuracy, datasetnumber):
    f = open("./exp1/" + modelname + "_" + datasetname + "_" + str(datasetnumber) + ".txt", "w")
    f.write("Model: " + modelname + "\n")
    f.write("Dataset " + str(datasetnumber) + ": " + datasetname + "\n")
    f.write("Accuracy: " + str(accuracy) + "\n")
    if accuracy!=None:
        f.write("@scores\n")
        for score in model.outlier_score:
            f.write(str(score) + "\n")
    f.close()

# This is based on executing the script from the folder experiments
ROUTE = "../datasets/outlier_ground_truth/"
#datasets = ["annthyroid.mat", "arrhythmia.mat", "breastw.mat", "cardio.mat", "cover.mat", "glass.mat", "ionosphere.mat", "letter.mat", "lympho.mat", "mammography.mat", "mnist.mat", "musk.mat", "optdigits.mat", "pendigits.mat", "pima.mat", "satellite.mat", "satimage-2.mat", "shuttle.mat", "speech.mat", "thyroid.mat", "vertebral.mat", "vowels.mat", "wbc.mat", "wine.mat"]
datasets = ["annthyroid.mat", "arrhythmia.mat", "breastw.mat", "cardio.mat", "glass.mat", "ionosphere.mat", "letter.mat", "lympho.mat", "mammography.mat", "mnist.mat", "musk.mat", "optdigits.mat", "pendigits.mat", "pima.mat", "satellite.mat", "satimage-2.mat", "speech.mat", "thyroid.mat", "vertebral.mat", "vowels.mat", "wbc.mat", "wine.mat"]

models = [TRINITY(verbose=True), KernelMahalanobis(), OUTRES(verbose=True), LODA(), HICS(verbose=True)]
names = ["TRINITY", "Mahalanobis Kernel", "OUTRES", "LODA", "HICS"]
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
            dataset, labels = readDataset(ROUTE + dat)
            print("The dataset has " + str(len(dataset)) + " number of instances with dimensionality " + str(len(dataset[0])))
            ker = fitModel(model, dataset)
            result = getAccuracy(model, labels, True)
        acc.append(result)
        if result==None:
            print("Accuracy: None")
        else:
            print("Accuracy: " + str(result*100) + "%")
        writeResults(name, dat, model, result, datasets.index(dat)+1)
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
