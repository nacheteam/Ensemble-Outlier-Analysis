#!# -*- coding: utf-8 -*-

import sys
sys.path.append('../models/')
from trinity import TRINITY

import numpy as np

# For reading the .mat files
import scipy.io

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
        if not np.nan in file['X'][i]:
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

'''
ker = TRINITY()
dataset, labels = readDataset("../datasets/outlier_ground_truth/cover.mat")
ker = fitModel(ker, dataset)
accuracy = getAccuracy(ker, labels, True)
print("The accuracy is " + str(accuracy*100) + "%")
'''
