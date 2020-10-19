from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.datasets import load_digits
from sklearn.linear_model import Perceptron
import numpy as np
from metrics import *
from utility import *

# Get Training Set and separate the features from classes
data_train2 = csvToList("train_2.csv")
X_train2, y_train2 = getFeaturesAndClass(data_train2)

# Get Validation set and separate the features from classes 
data_val2 = csvToList("val_2.csv")
X_val2, y_val2 = getFeaturesAndClass(data_val2)

# Running at default value which is 1000 iterations
clf = Perceptron(max_iter=1000, tol=1e-3)
clf.fit(X_train2, y_train2)

# Validation set
data_test2 = csvToList("test_with_label_2.csv")

# Separate features from classes
X_test2, y_test2 = getFeaturesAndClass(data_test2)
plotClassInstances(y_test2, 2, "Plotting of the test results for PER_2")

# Predict test values
y_test_pred2 = clf.predict(X_test2)
plotClassInstances(y_test_pred2, 2, "Plotting of the predicted results for PER_2")


createCSV("PER-DS2", y_test_pred2)


# Confusion Matrix
cm = confusion_matrix(y_test_pred2, y_test2)
print(cm)

# Metrics
precision, recall, f1, accuracy, f1_macro, f1_weight = getMetrics(y_test2, y_test_pred2)
writeMetrics("PER-DS2-Metrics", precision, recall, f1, accuracy, f1_macro, f1_weight)

# Getting Perceptron score
print(clf.score(X_train2, y_train2))

# Save confusion matrix as csv
np.savetxt("output\PER-DS2-Confusion_Matrix.csv", cm, delimiter=",", fmt='%s')