Experiments
===============

check_accuracy.py
--------------------

The code is available in: https://github.com/nacheteam/Ensemble-Outlier-Analysis/blob/master/experiments/check_accuracy.py

This file fits all models with all datasets.

The datasets used are available at ODDS: http://odds.cs.stonybrook.edu/

From this library the datasets used where:

============== ================= =======================
Name            Dimensionality   Number of instances
============== ================= =======================
  annthyroid         6                 7200
  arrhythmia        274                 452
   breastw           9                  683
    cardio           21                1831
    glass            9                  214
  ionosphere         33                 351
    letter           32                1600
    lympho           18                 148
 mammography         6                 11183
    mnist           100                7603
     musk           166                3062
  optdigits          64                5216
  pendigits          16                6870
     pima            8                  768
  satellite          36                6435
  satimage-2         36                5803
    speech          400                3686
   thyroid           6                 3772
  vertebral          6                  240
    vowels           12                1456
     wbc             30                 378
     wine            13                 129
============== ================= =======================

The models used are the 5 implemented by this module: HICS, OUTRES, LODA, Mahalanobis Kernel and TRINITY.

This file fits all the models with all datasets and saves the results in the files located under the folder exp1.
In this folder the files are splited in two, own containing our models and PyOD containing the PyOD models.

----

check_accuracy_PyOD.py
-------------------------------

This script implements the same functions adapted to PyOD modules.

The code can be found in: https://github.com/nacheteam/Ensemble-Outlier-Analysis/blob/master/experiments/check_accuracy_pyod.py

The models fitted in this script are: ABOD, COF, HBOS, KNN, LOF, MCD, OCSVM, PCA, SOD and SOS.

----

get_outlying_subspaces_outres.py
----------------------------------------

The implementation can be found in: https://github.com/nacheteam/Ensemble-Outlier-Analysis/blob/master/experiments/get_outlying_subspaces_outres.py

This script is the second main experiment of the repository. The purpose of the script is to execute OUTRES saving the internal information of each instance,
this is the subspaces where each instance is outlying and the adaptive neighborhood for that instance and that subspace. These results are saved in the folder exp2.

----

obtain_results.py
-------------------

The implementation can be found in: https://github.com/nacheteam/Ensemble-Outlier-Analysis/blob/master/experiments/obtain_results.py

This script takes the information from the first and second experiments and obtains results and figures from the data files.

The purpose of the script es reading the data from the files in exp1 and exp2 and plot bar figures with the time taken and accuracy obtained. The
script takes as well the files from exp2 and plot them individually.

----

obtain_auc_roc.py
------------------------

The implementation can be found in: https://github.com/nacheteam/Ensemble-Outlier-Analysis/blob/master/experiments/obtain_auc_roc.py

This script takes the results from the first experiments stored in the folder exp1. The script obtain the corresponding labels from each model and dataset and
obtains with them the AUC values and plots the ROC curves for our 5 models implemented.
