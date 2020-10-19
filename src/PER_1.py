from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.datasets import load_digits
from sklearn.linear_model import Perceptron
import numpy as np
from utility import *

# Get Training Set and separate the features from classes
data_train1 = csvToList("train_1.csv")
X_train1, y_train1 = getFeaturesAndClass(data_train1)

# Get Validation set and separate the features from classes 
data_val1 = csvToList("val_1.csv")
X_val1, y_val1 = getFeaturesAndClass(data_val1)

# Running at default value which is 1000 iterations
clf = Perceptron(max_iter=1000, tol=1e-3)
clf.fit(X_train1, y_train1)

# Validation set
data_test1 = csvToList("test_with_label_1.csv")

# Separate features from classes
X_test1, y_test1 = getFeaturesAndClass(data_test1)
plotClassInstances(y_test1, 1, "Plotting of the test results for PER_1")

# Predict test values
y_test_pred1 = clf.predict(X_test1)
plotClassInstances(y_test_pred1, 1, "Plotting of the predicted results for PER_1")


createCSV("PER-DS1", y_test_pred1)

# Confusion Matrix
cm = confusion_matrix(y_test_pred1, y_test1)
print(cm)

# Metrics
precision, recall, f1, accuracy, f1_macro, f1_weight = getMetrics(y_test1, y_test_pred1)
writeMetrics("PER-DS1-Metrics", precision, recall, f1, accuracy, f1_macro, f1_weight)

# Getting Perceptron Score
print(clf.score(X_train1, y_train1))

# Save confusion matrix as csv
np.savetxt("output\PER-DS1-Confusion_Matrix.csv", cm, delimiter=",", fmt='%s')