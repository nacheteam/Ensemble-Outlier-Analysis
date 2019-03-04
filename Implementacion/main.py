#!# -*- coding: utf-8 -*-
import numpy as np
from KernelMahalanobis import KernelMahalanobis
import matplotlib.pyplot as plt

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

################################################################################
##                                  Main                                      ##
################################################################################

def main():
    dataset = readData()

    # I place already the dataset as a matrix
    kernel_mahalanobis = KernelMahalanobis(NUM_ITERACIONES,dataset,len(dataset))
    scores = np.array(kernel_mahalanobis.runMethod())
    print("Media de los scores: " + str(np.mean(scores)))

    border = 0.9*(max(list(scores))-min(list(scores))) + min(list(scores))
    outliers = np.where(scores>=border)
    print("Los datos anómalos son los que tienen índices: " + str(list(outliers)))
    plt.scatter(list(range(len(scores))),scores,c="b",label="Datos no anómalos")
    plt.scatter(np.array(list(range(len(scores))))[outliers],scores[outliers],c="r",label="Anomalías")
    plt.xlabel("Dato")
    plt.ylabel("Score")
    plt.title("Scatter Plot de los scores, en rojo las anomalías")
    plt.legend()
    plt.show()
    # TODO: Plot the data seaborn-like for every pair of features
main()
