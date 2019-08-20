#!# -*- coding: utf-8 -*-

import sys
sys.path.append('../models/')
from trinity import TRINITY
from KernelMahalanobis import KernelMahalanobis
from hics import HICS
from loda import LODA
from outres import OUTRES

import numpy as np

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

# This is based on executing the script from the folder experiments
ROUTE = "../datasets/outlier_ground_truth/"
#datasets = ["annthyroid.mat", "arrhythmia.mat", "breastw.mat", "cardio.mat", "cover.mat", "glass.mat", "ionosphere.mat", "letter.mat", "lympho.mat", "mammography.mat", "mnist.mat", "musk.mat", "optdigits.mat", "pendigits.mat", "pima.mat", "satellite.mat", "satimage-2.mat", "shuttle.mat", "speech.mat", "thyroid.mat", "vertebral.mat", "vowels.mat", "wbc.mat", "wine.mat"]
datasets = ["annthyroid.mat", "arrhythmia.mat", "breastw.mat", "cardio.mat", "glass.mat", "ionosphere.mat", "letter.mat", "lympho.mat", "mammography.mat", "mnist.mat", "musk.mat", "optdigits.mat", "pendigits.mat", "pima.mat", "satellite.mat", "satimage-2.mat", "speech.mat", "thyroid.mat", "vertebral.mat", "vowels.mat", "wbc.mat", "wine.mat"]

models = [TRINITY(verbose=True), KernelMahalanobis(), HICS(verbose=True), OUTRES(verbose=True), LODA()]
names = ["TRINITY", "Mahalanobis Kernel", "HICS", "OUTRES", "LODA"]
accuracies = []

for name, model in zip(names, models):
    acc = []
    for dat in datasets:
        print("Computing dataset " + dat)
        dataset, labels = readDataset(ROUTE + dat)
        print("The dataset has " + str(len(dataset)) + " number of instances with dimensionality " + str(len(dataset[0])))
        ker = fitModel(model, dataset)
        acc.append(getAccuracy(model, labels, True))
    accuracies.append(acc)

print("\n\n\n")
for name, acc in zip(names, accuracies):
    print("\n\n#################################################################")
    print("MODEL " + name)
    print("#################################################################")
    for data, ac in zip(acc, datasets):
        print("The accuracy in the dataset " + data + " was " + str(ac*100) + "%")