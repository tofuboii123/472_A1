import numpy as np
from sklearn.metrics import *

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