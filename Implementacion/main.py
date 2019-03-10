#!# -*- coding: utf-8 -*-
import numpy as np
from KernelMahalanobis import KernelMahalanobis
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

NUM_ITERACIONES = 10
np.random.seed(12345)

################################################################################
##                      Función de lectura del dataset                        ##
################################################################################

def readData():
    data_file = open("./datasets/yeast/yeast.data")
    dataset = []
    classes = {"CYT":0,"NUC":1,"MIT":2,"ME3":3,"ME2":4,"ME1":5,"EXC":6,"VAC":7,"POX":8,"ERL":9}
    for line in data_file:
        # Removing que first column
        numerical = list(map(float,list(filter(None,line.split(" ")))[1:-1]))
        numerical.append(classes[line.split(" ")[-1].strip()])
        dataset.append(numerical)
    return np.matrix(dataset)

def createDataFrame(outliers):
    feature_names = ["mcg","gvh","alm","mit","erl","pox","vac","nuc"]
    data_file = open("./datasets/yeast/yeast.data")
    dataset = []
    i=0
    for line in data_file:
        # Removing the first column
        numerical = list(map(float,list(filter(None,line.split(" ")))[1:-1]))
        cl = line.split(" ")[-1].strip() if i not in outliers else "outlier"
        numerical.append(cl)
        dataset.append(numerical)
        i+=1
    return pd.DataFrame(data = dataset,columns = feature_names+["classes"])

def allPossiblePairs(list):
    pairs = []
    for i in range(len(list)):
        for j in range(i+1,len(list)):
            pairs.append([list[i],list[j]])
    return pairs

################################################################################
##                                  Main                                      ##
################################################################################

def main():
    dataset = readData()

    # I place already the dataset as a matrix
    kernel_mahalanobis = KernelMahalanobis(NUM_ITERACIONES,dataset,len(dataset))
    kernel_mahalanobis.runMethod()
    kernel_mahalanobis.obtainResults()

    # TODO: Plot the data seaborn-like for every pair of features
    data_frame = createDataFrame(kernel_mahalanobis.outliers)

    pairs = allPossiblePairs(["mcg","gvh","alm","mit","erl","pox","vac","nuc"])
    i=0
    for p in pairs:
        print("Pair " + str(i+1) + "/" + str(len(pairs)))
        i+=1
        sns.pairplot(data_frame,hue="classes",diag_kind="hist",vars=p,markers=9*["o"]+["D"]+["o"])
        plt.show()

main()
