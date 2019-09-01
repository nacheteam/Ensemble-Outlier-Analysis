#!# -*- coding: utf-8 -*-

import numpy as np
# Set the seed for the experiment
np.random.seed(123456789)

# Import the libraries
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import scipy.io
import sklearn.metrics

ROUTE = "./exp1/"

def readFileExp1(filename):
    '''
    @brief Function that reads the files from the first experiment
    @param filename Name of the file to obtain the information
    @return It returns a float with the accuracy between 0 and 1, the time
    in seconds and the scores as a list.
    '''
    file = open(filename, "r")
    accuracy = 0
    time = 0
    scores = []
    reading_scores = False
    for line in file:
        if reading_scores:
            if "[" in line:
                scores.append(float(line.split("[")[1].split("]")[0]))
            else:
                scores.append(float(line))
        if "Time taken" in line:
            time = float(line.split(":")[1].strip().split(" ")[0])
        elif "Accuracy" in line:
            accuracy = line.split(":")[1].strip()
            if accuracy=='None':
                accuracy = -1
            else:
                accuracy=float(accuracy)
        elif "@scores" in line:
            reading_scores = True
    return accuracy, time, np.array(scores)

def obtainLabels(scores,truelabels):
    unique, counts = np.unique(truelabels, return_counts=True)
    d = dict(zip(unique, counts))
    noutliers = d[1]
    outliers = []
    if noutliers<len(truelabels):
        outliers = scores.argsort()[-noutliers:][::-1]
    else:
        outliers = np.array(list(range(len(truelabels))))
    labels = np.array([0]*len(truelabels))
    for out in outliers:
        labels[out] = 1
    return labels

# Names of the files and the models
datasets = ["annthyroid.mat", "arrhythmia.mat", "breastw.mat", "cardio.mat", "glass.mat", "ionosphere.mat", "letter.mat", "lympho.mat", "mammography.mat", "mnist.mat", "musk.mat", "optdigits.mat", "pendigits.mat", "pima.mat", "satellite.mat", "satimage-2.mat", "speech.mat", "thyroid.mat", "vertebral.mat", "vowels.mat", "wbc.mat", "wine.mat"]
names = ["TRINITY", "Mahalanobis Kernel", "OUTRES", "LODA", "HICS", "ABOD", "COF", "HBOS", "KNN", "LOF", "MCD", "OCSVM", "PCA", "SOD", "SOS"]

# For each dataset
for dataset in datasets:
    # Load the dataset so we can plot it
    data = scipy.io.loadmat("../datasets/outlier_ground_truth/" + dataset)["X"]
    truelabels = scipy.io.loadmat("../datasets/outlier_ground_truth/" + dataset)["y"]
    # For each model name
    for name in names:
        if name in ["TRINITY", "Mahalanobis Kernel", "OUTRES", "LODA", "HICS"]:
            R = ROUTE + "own/"
        else:
            R = ROUTE + "pyod/"
        # Read the information of the file
        filename = R + name + "_" + dataset + "_" + str(datasets.index(dataset)+1) + ".txt"
        accuracy, time, scores = readFileExp1(filename)
        if accuracy!=-1 and name in ["TRINITY", "Mahalanobis Kernel", "OUTRES", "LODA", "HICS"]:
            prediction = obtainLabels(scores, truelabels)
            fpr, tpr, thr = sklearn.metrics.roc_curve(truelabels, prediction)
            auc = sklearn.metrics.auc(fpr,tpr)
            print("Modelo: " + name + ", dataset: " + dataset + ", AUC: " + str(auc))
            # Make a roc plot
            plt.title('Curva ROC')
            plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % auc)
            plt.legend(loc = 'lower right')
            plt.plot([0, 1], [0, 1],'r--')
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            plt.savefig("./imgs_exp1/roc/" + name + "_" + dataset.split(".")[0] + ".png")
            plt.close()
