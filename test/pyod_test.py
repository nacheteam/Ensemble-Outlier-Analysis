# Import all models that are going to be used
from pyod.models.abod import ABOD
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.cblof import CBLOF
from pyod.models.cof import COF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.loci import LOCI
from pyod.models.mcd import MCD
from pyod.models.mo_gaal import MO_GAAL
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.sod import SOD
from pyod.models.so_gaal import SO_GAAL
from pyod.models.sos import SOS

import sys, os

################################################################################
##                              AUX FUNCTIONS                                 ##
################################################################################

# Disable prints
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore prints
def enablePrint():
    sys.stdout = sys.__stdout__

################################################################################
##                        OUTLIERS FROM EACH MODEL                            ##
################################################################################

def getOutlierABOD(dataset):
    '''
    @brief Function that executes ABOD algorithm on the dataset and obtains the
    labels of the dataset indicating which instance is an inlier (0) or outlier (1)
    @param dataset Dataset on which to try the algorithm
    @return It returns a list of labels 0 means inlier, 1 means outlier
    '''
    # Initializating the model
    abod = ABOD()
    # Fits the data and obtains labels
    abod.fit(dataset)
    # Return labels
    return abod.labels_

def getOutlierAutoEncoder(dataset):
    '''
    @brief Function that executes AutoEncoder algorithm on the dataset and obtains the
    labels of the dataset indicating which instance is an inlier (0) or outlier (1)
    @param dataset Dataset on which to try the algorithm
    @return It returns a list of labels 0 means inlier, 1 means outlier
    '''
    # Initializating the model with 3 layers of neurons and 8, 6, 8 neurons per layer without verbose
    ae = AutoEncoder(hidden_neurons=[8,6,8],verbose=0)
    # Fits the data and obtains labels
    ae.fit(dataset)
    # Return labels
    return ae.labels_

def getOutlierCBLOF(dataset):
    '''
    @brief Function that executes CBLOF algorithm on the dataset and obtains the
    labels of the dataset indicating which instance is an inlier (0) or outlier (1)
    @param dataset Dataset on which to try the algorithm
    @return It returns a list of labels 0 means inlier, 1 means outlier
    '''
    # Initializating the model
    cblof = CBLOF()
    # Fits the data and obtains labels
    cblof.fit(dataset)
    # Return labels
    return cblof.labels_

def getOutlierCOF(dataset):
    '''
    @brief Function that executes COF algorithm on the dataset and obtains the
    labels of the dataset indicating which instance is an inlier (0) or outlier (1)
    @param dataset Dataset on which to try the algorithm
    @return It returns a list of labels 0 means inlier, 1 means outlier
    '''
    # Initializating the model
    cof = COF()
    # Fits the data and obtains labels
    cof.fit(dataset)
    # Return labels
    return cof.labels_

def getOulierFeatureBagging(dataset):
    '''
    @brief Function that executes Feature Bagging algorithm on the dataset and obtains the
    labels of the dataset indicating which instance is an inlier (0) or outlier (1)
    @param dataset Dataset on which to try the algorithm
    @return It returns a list of labels 0 means inlier, 1 means outlier
    '''
    # Initializating the model without verbose
    fb = FeatureBagging(verbose=0)
    # Fits the data and obtains labels
    fb.fit(dataset)
    # Return labels
    return fb.labels_

def getOutlierHBOS(dataset):
    '''
    @brief Function that executes HBOS algorithm on the dataset and obtains the
    labels of the dataset indicating which instance is an inlier (0) or outlier (1)
    @param dataset Dataset on which to try the algorithm
    @return It returns a list of labels 0 means inlier, 1 means outlier
    '''
    # Initializating the model
    hbos = HBOS()
    # Fits the data and obtains labels
    hbos.fit(dataset)
    # Return labels
    return hbos.labels_

def getOutlierIForest(dataset):
    '''
    @brief Function that executes IForest algorithm on the dataset and obtains the
    labels of the dataset indicating which instance is an inlier (0) or outlier (1)
    @param dataset Dataset on which to try the algorithm
    @return It returns a list of labels 0 means inlier, 1 means outlier
    '''
    # Initializating the model without verbose
    ifor = IForest(verbose=0)
    # Fits the data and obtains labels
    ifor.fit(dataset)
    # Return labels
    return ifor.labels_

def getOutlierKNN(dataset):
    '''
    @brief Function that executes KNN algorithm on the dataset and obtains the
    labels of the dataset indicating which instance is an inlier (0) or outlier (1)
    @param dataset Dataset on which to try the algorithm
    @return It returns a list of labels 0 means inlier, 1 means outlier
    '''
    # Initializating the model
    knn = KNN()
    # Fits the data and obtains labels
    knn.fit(dataset)
    # Return labels
    return knn.labels_

def getOutlierLOF(dataset):
    '''
    @brief Function that executes LOF algorithm on the dataset and obtains the
    labels of the dataset indicating which instance is an inlier (0) or outlier (1)
    @param dataset Dataset on which to try the algorithm
    @return It returns a list of labels 0 means inlier, 1 means outlier
    '''
    # Initializating the model
    lof = LOF()
    # Fits the data and obtains labels
    lof.fit(dataset)
    # Return labels
    return lof.labels_

def getOutlierMCD(dataset):
    '''
    @brief Function that executes MCD algorithm on the dataset and obtains the
    labels of the dataset indicating which instance is an inlier (0) or outlier (1)
    @param dataset Dataset on which to try the algorithm
    @return It returns a list of labels 0 means inlier, 1 means outlier
    '''
    # Initializating the model
    mcd = MCD()
    # Fits the data and obtains labels
    mcd.fit(dataset)
    # Return labels
    return mcd.labels_

def getOulierMOGAAL(dataset):
    '''
    @brief Function that executes MO_GAAL algorithm on the dataset and obtains the
    labels of the dataset indicating which instance is an inlier (0) or outlier (1)
    @param dataset Dataset on which to try the algorithm
    @return It returns a list of labels 0 means inlier, 1 means outlier
    '''
    # Initializating the model
    mg = MO_GAAL()
    # Fits the data and obtains labels
    mg.fit(dataset)
    # Return labels
    return mg.labels_

def getOutlierOCSVM(dataset):
    '''
    @brief Function that executes OCSVM algorithm on the dataset and obtains the
    labels of the dataset indicating which instance is an inlier (0) or outlier (1)
    @param dataset Dataset on which to try the algorithm
    @return It returns a list of labels 0 means inlier, 1 means outlier
    '''
    # Initializating the model
    ocsvm = OCSVM()
    # Fits the data and obtains labels
    ocsvm.fit(dataset)
    # Return labels
    return ocsvm.labels_

def getOutlierPCA(dataset):
    '''
    @brief Function that executes PCA algorithm on the dataset and obtains the
    labels of the dataset indicating which instance is an inlier (0) or outlier (1)
    @param dataset Dataset on which to try the algorithm
    @return It returns a list of labels 0 means inlier, 1 means outlier
    '''
    # Initializating the model
    pca = PCA()
    # Fits the data and obtains labels
    pca.fit(dataset)
    # Return labels
    return pca.labels_

def getOutlierSOD(dataset):
    '''
    @brief Function that executes SOD algorithm on the dataset and obtains the
    labels of the dataset indicating which instance is an inlier (0) or outlier (1)
    @param dataset Dataset on which to try the algorithm
    @return It returns a list of labels 0 means inlier, 1 means outlier
    '''
    # Initializating the model
    sod = SOD()
    # Fits the data and obtains labels
    sod.fit(dataset)
    # Return labels
    return sod.labels_

def getOutlierSOGAAL(dataset):
    '''
    @brief Function that executes SO_GAAL algorithm on the dataset and obtains the
    labels of the dataset indicating which instance is an inlier (0) or outlier (1)
    @param dataset Dataset on which to try the algorithm
    @return It returns a list of labels 0 means inlier, 1 means outlier
    '''
    # Initializating the model
    sg = SO_GAAL()
    # Fits the data and obtains labels
    sg.fit(dataset)
    # Return labels
    return sg.labels_

def getOulierSOS(dataset):
    '''
    @brief Function that executes SOS algorithm on the dataset and obtains the
    labels of the dataset indicating which instance is an inlier (0) or outlier (1)
    @param dataset Dataset on which to try the algorithm
    @return It returns a list of labels 0 means inlier, 1 means outlier
    '''
    # Initializating the model
    sos = SOS()
    # Fits the data and obtains labels
    sos.fit(dataset)
    # Return labels
    return sos.labels_

################################################################################
##                           MAIN FUNCTIONALITY                               ##
################################################################################

def voteForOutliers(dataset, n_votes=5):
    '''
    @brief Function that executes all algorithms described above for outlier detection
    on the dataset and obtains the labels for each algorithm couting the number of times an
    instance is classified as outlier. If it get more than n_votes times classified
    as outlier then we put this index as "real outlier".
    @param dataset Dataset on which to apply the voting system
    @param n_votes Minimum number of times that an instance needs to be classified
    as an outlier for it to be considered as ground truth.
    @return It returns a list of indexes that match the index of the instance in the
    dataset classified as real outlier.
    '''
    # Names of the functions
    models = [getOutlierABOD, getOutlierAutoEncoder, getOutlierCBLOF, getOutlierCOF, getOulierFeatureBagging, getOutlierHBOS, getOutlierIForest, getOutlierKNN, getOutlierLOF, getOutlierMCD, getOulierMOGAAL, getOutlierOCSVM, getOutlierPCA, getOutlierSOD, getOutlierSOGAAL, getOulierSOS]
    # Names of the models
    names = ["ABOD", "Auto Encoder", "CBLOF", "COF", "Feature Bagging", "HBOS", "IForest", "KNN", "LOF", "MCD", "MO_GAAL", "OCSVM", "PCA", "SOD", "SO_GAAL", "SOS"]
    # Couning list for each instance, declared first as zeros
    countings = [0]*len(dataset)
    # For each model and name
    for model,name in zip(models,names):
        enablePrint()
        print("Executing " + name)
        blockPrint()
        # Execute the model
        outliers = model(dataset)
        # Update the countings
        for i in range(len(outliers)):
            if outliers[i]==1:
                countings[i]+=1
    # Voted outliers are the instances with at least n_votes times being classified as outlier
    voted_outliers = []
    for i in range(len(countings)):
        if countings[i]>=n_votes:
            voted_outliers.append(i)
    enablePrint()
    return voted_outliers
