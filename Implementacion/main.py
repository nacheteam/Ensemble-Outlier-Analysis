#!# -*- coding: utf-8 -*-
import numpy as np
from KernelMahalanobis import KernelMahalanobis
import matplotlib.pyplot as plt

data_file = open("./datasets/yeast/yeast.data")
dataset = []
for line in data_file:
    # Removing que first and last column
    dataset.append(list(map(float,list(filter(None,line.split(" ")))[1:-1])))

# I place already the dataset as a matrix
kernel_mahalanobis = KernelMahalanobis(10,np.matrix(dataset),len(dataset))
scores = np.array(kernel_mahalanobis.runMethod())
print("Media de los scores: " + str(np.mean(scores)))

border = 0.9*(max(list(scores))-min(list(scores))) + min(list(scores))
outliers = np.where(scores>=border)
plt.scatter(list(range(len(scores))),scores,c="b")
plt.scatter(np.array(list(range(len(scores))))[outliers],scores[outliers],c="r")
plt.xlabel("Dato")
plt.ylabel("Score")
plt.title("Scatter Plot de los scores, en rojo las anomalías")
plt.legend()
plt.show()
