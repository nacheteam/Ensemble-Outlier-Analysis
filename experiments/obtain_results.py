#!# -*- coding: utf-8 -*-

import numpy as np
# Set the seed for the experiment
np.random.seed(123456789)

import matplotlib.pyplot as plt

ROUTE = "./exp1/"

def readFileExp1(filename):
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

datasets = ["annthyroid.mat", "arrhythmia.mat", "breastw.mat", "cardio.mat", "glass.mat", "ionosphere.mat", "letter.mat", "lympho.mat", "mammography.mat", "mnist.mat", "musk.mat", "optdigits.mat", "pendigits.mat", "pima.mat", "satellite.mat", "satimage-2.mat", "speech.mat", "thyroid.mat", "vertebral.mat", "vowels.mat", "wbc.mat", "wine.mat"]
names = ["TRINITY", "Mahalanobis Kernel", "OUTRES", "LODA", "HICS", "ABOD", "COF", "HBOS", "KNN", "LOF", "MCD", "OCSVM", "PCA", "SOD", "SOS"]

for dataset in datasets:
    accuracies = []
    labels = []
    for name in names:
        if name in ["TRINITY", "Mahalanobis Kernel", "OUTRES", "LODA", "HICS"]:
            R = ROUTE + "own/"
        else:
            R = ROUTE + "pyod/"
        filename = R + name + "_" + dataset + "_" + str(datasets.index(dataset)+1) + ".txt"
        accuracy, time, scores = readFileExp1(filename)
        if accuracy!=-1:
            accuracies.append(accuracy*100)
            labels.append(name)
    index = np.arange(len(labels))
    plt.bar(index,accuracies)
    plt.xlabel("Modelos", fontsize=10)
    plt.ylabel("Porcentaje de acierto", fontsize=10)
    plt.xticks(index, labels, fontsize=7, rotation=60)
    plt.title("Porcentaje de acierto en el conjunto " + dataset)
    plt.savefig("./imgs_exp1/accuracy/" + dataset.split(".")[0] + ".png")
    plt.close()

for dataset in datasets:
    times = []
    labels = []
    for name in names:
        if name in ["TRINITY", "Mahalanobis Kernel", "OUTRES", "LODA", "HICS"]:
            R = ROUTE + "own/"
        else:
            R = ROUTE + "pyod/"
        filename = R + name + "_" + dataset + "_" + str(datasets.index(dataset)+1) + ".txt"
        accuracy, time, scores = readFileExp1(filename)
        if accuracy!=-1:
            times.append(time)
            labels.append(name)
    index = np.arange(len(labels))
    plt.bar(index,times)
    plt.xlabel("Modelos", fontsize=10)
    plt.ylabel("Tiempo en segundos", fontsize=10)
    plt.xticks(index, labels, fontsize=7, rotation=60)
    plt.title("Tiempo de cómputo de los modelos en " + dataset)
    plt.savefig("./imgs_exp1/times/" + dataset.split(".")[0] + ".png")
    plt.close()
