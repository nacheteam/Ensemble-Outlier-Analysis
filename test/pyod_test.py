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
from pyod.models.lscp import LSCP
from pyod.models.mcd import MCD
from pyod.models.mo_gaal import MO_GAAL
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.sod import SOD
from pyod.models.so_gaal import SO_GAAL
from pyod.models.sos import SOS
from pyod.models.xgbod import XGBOD

################################################################################
##                        OUTLIERS FROM EACH MODEL                            ##
################################################################################

def getOutlierABOD(dataset):
    abod = ABOD()
    abod.fit(dataset)
    return abod.labels_

def getOutlierAutoEncoder(dataset):
    ae = AutoEncoder()
    ae.fit(dataset)
    return ae.labels_

def getOutlierCBLOF(dataset):
    cblof = CBLOF()
    cblof.fit(dataset)
    return cblof.labels_

def getOutlierCOF(dataset):
    cof = COF()
    cof.fit(dataset)
    return cof.labels_

def getOulierFeatureBagging(dataset):
    fb = FeatureBagging()
    fb.fit(dataset)
    return fb.labels_

def getOutlierHBOS(dataset):
    hbos = HBOS()
    hbos.fit(dataset)
    return hbos.labels_

def getOutlierIForest(dataset):
    ifor = IForest()
    ifor.fit(dataset)
    return ifor.labels_

def getOutlierKNN(dataset):
    knn = KNN()
    knn.fit(dataset)
    return knn.labels_

def getOutlierLOF(dataset):
    lof = LOF()
    lof.fit(dataset)
    return lof.labels_

def getOutlierLOCI(dataset):
    loci = LOCI()
    loci.fit(dataset)
    return loci.labels_

def getOutlierLSCP(dataset):
    lscp = LSCP()
    lscp.fit(dataset)
    return lscp.labels_

def getOutlierMCD(dataset):
    mcd = MCD()
    mcd.fit(dataset)
    return mcd.labels_

def getOulierMOGAAL(dataset):
    mg = MO_GAAL()
    mg.fit(dataset)
    return mg.labels_

def getOutlierOCSVM(dataset):
    ocsvm = OCSVM()
    ocsvm.fit(dataset)
    return ocsvm.labels_

def getOutlierPCA(dataset):
    pca = PCA()
    pca.fit(dataset)
    return pca.labels_

def getOutlierSOD(dataset):
    sod = SOD()
    sod.fit(dataset)
    return sod.labels_

def getOutlierSOGAAL(dataset):
    sg = SO_GAAL()
    sg.fit(dataset)
    return sg.labels_

def getOulierSOS(dataset):
    sos = SOS()
    sos.fit(dataset)
    return sos.labels_

def getOulierXGBOD(dataset):
    xgbod = XGBOD()
    xgbod.fit(dataset)
    return xgbod.labels_

################################################################################
##                           MAIN FUNCTIONALITY                               ##
################################################################################

def voteForOutliers(dataset, n_votes=5):
    models = [getOutlierABOD, getOutlierAutoEncoder, getOutlierCBLOF, getOutlierCOF, getOulierFeatureBagging, getOutlierHBOS, getOutlierIForest, getOutlierKNN, getOutlierLOF, getOutlierLOCI, getOutlierLSCP, getOutlierMCD, getOulierMOGAAL, getOutlierOCSVM, getOutlierPCA, getOutlierSOD, getOutlierSOGAAL, getOulierSOS, getOulierXGBOD]
    countings = [0]*len(dataset)
    for model in models:
        outliers = model(dataset)
        for i in range(len(outliers)):
            if outliers[i]==1:
                countings[i]+=1
    voted_outliers = []
    for i in range(len(countings)):
        if countings[i]>=n_votes:
            voted_outliers.append(i)
    return voted_outliers