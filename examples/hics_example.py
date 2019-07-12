#!# -*- coding: utf-8 -*-
import sys
sys.path.append('../models/')
sys.path.append("../test/")

import numpy as np
from hics import HICS
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
    #dataset,labels = utils.readDataAbalone()
    dataset,labels = utils.readDataYeast()

    hics = HICS(verbose=True)
    hics.fit(dataset)
    utils.obtainResults(hics)

    outliers = hics.getOutliers()

    print(labels[outliers])

    print("Getting anomalies based on the voting system to check")
    cm, df = utils.checkAnomalies(dataset, outliers)
    print("Common ones: " + str(cm))
    print("Different ones: " + str(df))

main()
