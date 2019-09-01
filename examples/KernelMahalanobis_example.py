#!# -*- coding: utf-8 -*-
import sys
# We add the path to the models
sys.path.append('../models/')

# Import all used libraries
import numpy as np
from KernelMahalanobis import KernelMahalanobis
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import utils

from pyod_test import voteForOutliers

np.random.seed(123456789)

################################################################################
##                                  Main                                      ##
################################################################################

def main():
    # Read the data
    #dataset,labels = utils.readDataAbalone()
    dataset,labels = utils.readDataYeast()

    # I place already the dataset as a matrix
    kernel_mahalanobis = KernelMahalanobis()
    kernel_mahalanobis.fit(dataset)
    utils.obtainResults(kernel_mahalanobis)

    # Get the outliers
    outliers = kernel_mahalanobis.getOutliers()

    # Print the labels of the outliers
    print(labels[outliers])

    # Check if pyod models get the same anomalies
    print("Getting anomalies based on the voting system to check")
    cm, df = utils.checkAnomalies(dataset, outliers)
    print("Common ones: " + str(cm))
    print("Different ones: " + str(df))

main()
