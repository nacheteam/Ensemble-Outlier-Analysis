#!# -*- coding: utf-8 -*-
import numpy as np
from KernelMahalanobis import KernelMahalanobis

data_file = open("./datasets/yeast/yeast.data")
dataset = []
for line in data_file:
    # Removing que first and last column
    dataset.append(list(map(float,list(filter(None,line.split(" ")))[1:-1])))

kernel_mahalanobis = KernelMahalanobis(10,np.array(dataset),len(dataset))
kernel_mahalanobis.runMethod()
