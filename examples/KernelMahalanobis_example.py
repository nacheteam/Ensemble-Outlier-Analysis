#!# -*- coding: utf-8 -*-
import sys
sys.path.append('../')

import numpy as np
from KernelMahalanobis import KernelMahalanobis
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

NUM_ITERACIONES = 100
np.random.seed(123456789)

################################################################################
##                      Función de lectura del dataset                        ##
################################################################################

def readData():
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

################################################################################
##                                  Main                                      ##
################################################################################

def main():
    dataset,labels = readData()

    # I place already the dataset as a matrix
    kernel_mahalanobis = KernelMahalanobis(NUM_ITERACIONES,dataset,len(dataset))
    kernel_mahalanobis.runMethodClassic()
    kernel_mahalanobis.obtainResults()

    print(labels[kernel_mahalanobis.outliers])

main()
