import numpy as np
from sklearn.metrics import *
import matplotlib.pyplot as plt

def csvToList(csv):
    return np.genfromtxt(f"Assig1-Dataset/{csv}", delimiter=",")

def getFeaturesAndClass(data):
    return data[:, :-1], data[:,-1]

def getMetrics(y_true, y_pred):
    precision = precision_score(y_true, y_pred, average=None)
    print("Precision:", precision)
    recall = recall_score(y_true, y_pred, average=None)
    print("Recall:", recall)
    f1 = f1_score(y_true, y_pred, average=None)
    print("f1 score:", f1)

    accuracy = accuracy_score(y_true, y_pred)
    print("Accuracy:", accuracy)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    print("f1 macro:", f1_macro)
    f1_weight = f1_score(y_true, y_pred, average="weighted")
    print("f1 weight:", f1_weight)

    return precision, recall, f1, accuracy, f1_macro, f1_weight

def createCSV(name, y_pred):
    f = open(f"output/{name}.csv", "w")

    f.write("instance,predicted_class\n")

    for i in range(len(y_pred)):
        f.write(f"{i + 1},{int(y_pred[i])}\n")
    
    f.close()

def writeMetrics(name, precision, recall, f1, accuracy, f1_macro, f1_weight):
    f = open(f"output/{name}.csv", "w")

    f.write("class,precision,recall,f1\n")

    assert(len(precision) == len(recall) and len(recall) == len(f1))

    for i in range(len(precision)):
        f.write(f"{i},{precision[i]},{recall[i]},{f1[i]}\n")

    f.write("\naccuracy,f1_macro,f1_weight\n")
    f.write(f"{accuracy},{f1_macro},{f1_weight}")
    f.close()

def plotClassInstances(predicted_labels, dataset, title):
    if dataset == 1:
        labels = range(0,26)
    if dataset == 2:
        labels = range(0,10)
    num_occurences = count_labels(labels, predicted_labels)
    plt.bar(labels, num_occurences, align='center', width=0.5)
    plt.gca().set_xticks(labels)
    value_onTop(num_occurences)
    plt.xlabel("Labels")
    plt.ylabel("Number of times occured")
    plt.title(title)
    plt.show()

def count_labels(labels,predicted_labels):
    num_occurences = []
    for value in labels:
        num_occurences.append(np.count_nonzero(predicted_labels == value))
    return num_occurences

def value_onTop(num_occurences):
    for index, value in enumerate(num_occurences):
        plt.text(index - 0.25, value + 0.01, str(value))
        

