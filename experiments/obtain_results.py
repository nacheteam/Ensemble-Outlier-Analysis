#!# -*- coding: utf-8 -*-

import numpy as np
# Set the seed for the experiment
np.random.seed(123456789)

# Import the libraries
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import scipy.io

ROUTE = "./exp1/"

def readFileExp1(filename):
    '''
    @brief Function that reads the files from the first experiment
    @param filename Name of the file to obtain the information
    @return It returns a float with the accuracy between 0 and 1, the time
    in seconds and the scores as a list.
    '''
    file = open(filename, "r")
    accuracy = 0
    time = 0
    scores = []
    reading_scores = False
    for line in file:
        if reading_scores:
            if "[" in line:
                scores.append(float(line.split("[")[1].split("]")[0]))
            else:
                scores.append(float(line))
        if "Time taken" in line:
            time = float(line.split(":")[1].strip().split(" ")[0])
        elif "Accuracy" in line:
            accuracy = line.split(":")[1].strip()
            if accuracy=='None':
                accuracy = -1
            else:
                accuracy=float(accuracy)
        elif "@scores" in line:
            reading_scores = True
    return accuracy, time, scores

# Names of the files and the models
datasets = ["annthyroid.mat", "arrhythmia.mat", "breastw.mat", "cardio.mat", "glass.mat", "ionosphere.mat", "letter.mat", "lympho.mat", "mammography.mat", "mnist.mat", "musk.mat", "optdigits.mat", "pendigits.mat", "pima.mat", "satellite.mat", "satimage-2.mat", "speech.mat", "thyroid.mat", "vertebral.mat", "vowels.mat", "wbc.mat", "wine.mat"]
names = ["TRINITY", "Mahalanobis Kernel", "OUTRES", "LODA", "HICS", "ABOD", "COF", "HBOS", "KNN", "LOF", "MCD", "OCSVM", "PCA", "SOD", "SOS"]

# For each dataset
for dataset in datasets:
    accuracies = []
    labels = []
    # For each model name
    for name in names:
        if name in ["TRINITY", "Mahalanobis Kernel", "OUTRES", "LODA", "HICS"]:
            R = ROUTE + "own/"
        else:
            R = ROUTE + "pyod/"
        # Read the information of the file
        filename = R + name + "_" + dataset + "_" + str(datasets.index(dataset)+1) + ".txt"
        accuracy, time, scores = readFileExp1(filename)
        if accuracy!=-1:
            accuracies.append(accuracy*100)
            labels.append(name)
    # Make a bar plot of the accuracies
    index = np.arange(len(labels))
    plt.bar(index,accuracies)
    plt.xlabel("Modelos", fontsize=10)
    plt.ylabel("Porcentaje de acierto", fontsize=10)
    plt.xticks(index, labels, fontsize=7, rotation=60)
    plt.title("Porcentaje de acierto en el conjunto " + dataset)
    plt.savefig("./imgs_exp1/accuracy/" + dataset.split(".")[0] + ".png")
    plt.close()

# For each dataset
for dataset in datasets:
    times = []
    labels = []
    # For each model name
    for name in names:
        if name in ["TRINITY", "Mahalanobis Kernel", "OUTRES", "LODA", "HICS"]:
            R = ROUTE + "own/"
        else:
            R = ROUTE + "pyod/"
        # Read the file
        filename = R + name + "_" + dataset + "_" + str(datasets.index(dataset)+1) + ".txt"
        accuracy, time, scores = readFileExp1(filename)
        if accuracy!=-1:
            times.append(time)
            labels.append(name)
    # Make a bar plot of the times
    index = np.arange(len(labels))
    plt.bar(index,times)
    plt.xlabel("Modelos", fontsize=10)
    plt.ylabel("Tiempo en segundos", fontsize=10)
    plt.xticks(index, labels, fontsize=7, rotation=60)
    plt.title("Tiempo de cómputo de los modelos en " + dataset)
    plt.savefig("./imgs_exp1/times/" + dataset.split(".")[0] + ".png")
    plt.close()


# Take the outres pima exec
file = "./exp2/outres_pima.mat_6.txt"
f = open(file, "r")
outliers = []
subspaces = []
neighborhoods = []
reading_subspaces = False
# Read the outliers, subspaces and neighborhoods
for line in f:
    if reading_subspaces:
        out = int(line.split(";")[0])
        sub = []
        nei = []
        for s in line.split(";")[1].split(","):
            sub.append(int(s))
        for n in line.split(";")[2].split(","):
            nei.append(int(n))
        outliers.append(out)
        subspaces.append(sub)
        neighborhoods.append(nei)
    if "outlier ; subspace ; neighborhood" in line:
        reading_subspaces=True

# Load the dataset so we can plot it
data = scipy.io.loadmat("../datasets/outlier_ground_truth/pima.mat")["X"]
labels = scipy.io.loadmat("../datasets/outlier_ground_truth/pima.mat")["y"]

cont = 0
# For each outlier, each subspace ant the neighborhood
for out, sub, neig in zip(outliers, subspaces, neighborhoods):
    cont+=1
    print("Obtaining image " + str(cont) + "/" + str(len(outliers)))
    # If it has dimension 2
    if len(sub)==2:
        # We plot directly the data
        proj = data[:,sub]
        plt.scatter(proj[:,0][out], proj[:,1][out], label="Instancia", c="red")
        plt.scatter(proj[:,0][neig], proj[:,1][neig], label="Vecindario", c="blue")
        plt.title("Instancia " + str(out) + " en el subespacio " + str(sub))
        plt.legend()
        plt.savefig("./imgs_exp2/" + str(cont) + ".png")
        plt.close()
    # If the dimensionality is bigger than 2
    else:
        # First we obtain the TSNE projection
        proj = data[:,sub]
        tsne_proj = TSNE(n_components=2).fit_transform(proj)
        plt.scatter(tsne_proj[:,0][out], tsne_proj[:,1][out], label="Instancia", c="red")
        plt.scatter(tsne_proj[:,0][neig], tsne_proj[:,1][neig], label="Vecindario", c="blue")
        plt.title("Instancia " + str(out) + " en el subespacio " + str(sub))
        plt.legend()
        plt.savefig("./imgs_exp2/" + str(cont) + "_tsne.png")
        plt.close()
