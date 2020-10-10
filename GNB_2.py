from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
import numpy as np
from metrics import *
from utility import *

# Training data set
data_train2 = csvToList("train_2.csv")
# Separate features from classes
X_train1, y_train1 = getFeaturesAndClass(data_train2)

# Validation set
data_val2 = csvToList("val_2.csv")

# Separate features from classes
X_val1, y_val1 = getFeaturesAndClass(data_val2)

# Apply GNB model
gnb = GaussianNB()
y_val_pred1 = gnb.fit(X_train1, y_train1).predict(X_val1)

cm = confusion_matrix(y_val1, y_val_pred1)
# print(cm)
print((y_val1 != y_val_pred1).sum())

# Validation set
data_test2 = csvToList("test_with_label_2.csv")

# Separate features from classes
X_test1, y_test1 = getFeaturesAndClass(data_test2)


# Predict test values
y_test_pred1 = gnb.fit(X_train1, y_train1).predict(X_test1)
cm_test = confusion_matrix(y_test_pred1, y_test1)
print((y_test1 != y_test_pred1).sum())
print(cm_test)

# Metrics
precision(cm_test)
recall(cm_test)
f1_measure(cm_test)