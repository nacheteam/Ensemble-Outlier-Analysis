#!# -*- coding: utf-8 -*-
import sys
sys.path.append('../models/')
sys.path.append("../test/")

import numpy as np
from KernelMahalanobis import KernelMahalanobis
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from pyod_test import voteForOutliers

NUM_ITERACIONES = 100
NOUTLIERS=10
np.random.seed(123456789)

################################################################################
##                      Función de lectura del dataset                        ##
################################################################################

def readDataAbalone():
    data_file = open("../datasets/abalone19.dat")
    dataset = []
    labels = []
    # To parse classes and sex column to numerical
    classes = {"negative":0,"positive":1}
    sex = {"M":0, "F":1, "I":2}
    reading_data=False
    for line in data_file:
        if reading_data:
            vars = line.rstrip().split(",")
            row=[]
            for i in range(len(vars)):
                if i==0:
                    row.append(sex[vars[i].strip()])
                elif i==len(vars)-1:
                    labels.append(classes[vars[i].strip()])
                else:
                    row.append(float(vars[i]))
            dataset.append(row)
        if "@data" in line and not reading_data:
            reading_data = True
    return np.matrix(dataset), np.array(labels)

def readDataYeast():
    data_file = open("../datasets/yeast6.dat")
    dataset = []
    labels = []
    # To parse classes to numerical
    classes = {"negative":0,"positive":1}
    reading_data=False
    for line in data_file:
        if reading_data:
            vars = line.rstrip().split(",")
            row=[]
            for i in range(len(vars)):
                if i==len(vars)-1:
                    labels.append(classes[vars[i].strip()])
                else:
                    row.append(float(vars[i]))
            dataset.append(row)
        if "@data" in line and not reading_data:
            reading_data = True
    return np.matrix(dataset), np.array(labels)

def obtainResults(mk):
    print("\n\n\n##########################################################")
    print("Statistical information about the outliers")
    print("##########################################################\n\n")
    scores = mk.getRawScores()
    # First the mean, stdv and variance
    mean = np.mean(scores)
    stdv = np.std(scores)
    variance = np.var(scores)
    print("The mean of the scores is: " + str(mean))
    print("The standard deviation of the scores is: " + str(stdv))
    print("The variance of the scores is: " + str(variance))

    outliers = mk.getOutliersBN(NOUTLIERS)
    print("The outliers are the elements with indexes: " + str(outliers))

    # Then we will plot the scores colouring the outliers in red.
    plt.scatter(list(range(len(scores))),scores,c="b",label="Normal data")
    plt.scatter(np.array(list(range(len(scores))))[outliers],scores[outliers],c="r",label="Outliers")
    plt.xlabel("Data")
    plt.ylabel("Score")
    plt.title("Scatter Plot of the scores, outliers in red.")
    plt.legend()
    plt.show()

def checkAnomalies(dataset, outliers_mahalanobis):
    outliers_voted = voteForOutliers(dataset)
    common_ones = 0
    different_ones = 0
    for om in outliers_mahalanobis:
        if om in outliers_voted:
            common_ones+=1
        else:
            different_ones+=1
    return common_ones, different_ones

################################################################################
##                                  Main                                      ##
################################################################################

def main():
    #dataset,labels = readDataAbalone()
    dataset,labels = readDataYeast()

    # I place already the dataset as a matrix
    kernel_mahalanobis = KernelMahalanobis(NUM_ITERACIONES)
    kernel_mahalanobis.fit(dataset)
    kernel_mahalanobis.runMethod()
    obtainResults(kernel_mahalanobis)

    print(labels[kernel_mahalanobis.getOutliersBN(NOUTLIERS)])

    cm, df = checkAnomalies(dataset, kernel_mahalanobis.getOutliersBN(NOUTLIERS))
    print("Common ones: " + str(cm))
    print("Different ones: " + str(df))

main()
