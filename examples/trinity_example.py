#!# -*- coding: utf-8 -*-
import sys
sys.path.append('../models/')
sys.path.append("../test/")

import numpy as np
from trinity import TRINITY
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import utils

from pyod_test import voteForOutliers

from cProfile import Profile
import pstats, io

np.random.seed(123456789)

################################################################################
##                                  Main                                      ##
################################################################################

def main():
    #dataset,labels = utils.readDataAbalone()
    dataset,labels = utils.readDataYeast()
    #dataset, labels = utils.readDataCancer()

    trinity = TRINITY(verbose=True)
    '''
    p = Profile()
    p.enable()
    '''
    trinity.fit(dataset)
    '''
    p.disable()
    s = io.StringIO()
    ps = pstats.Stats(p, stream=s).sort_stats('cumulative')
    ps.print_stats()
    print(s.getvalue())
    utils.obtainResults(outres)
    '''
    outliers = trinity.getOutliers()

    print(labels[outliers])

    print("Getting anomalies based on the voting system to check")
    cm, df = utils.checkAnomalies(dataset, outliers)
    print("Common ones: " + str(cm))
    print("Different ones: " + str(df))

main()
