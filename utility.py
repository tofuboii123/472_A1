import numpy as np

def csvToList(csv):
    return np.genfromtxt(f"Assig1-Dataset/{csv}", delimiter=",")

def getFeaturesAndClass(data):
    return data[:, :-1], data[:,-1]