from sklearn import tree
from sklearn.tree import export_text
from sklearn.metrics import confusion_matrix
import numpy as np
from metrics import *
from utility import *

# Training set
data_train2 = csvToList("train_2.csv")
X_train1, y_train1 = getFeaturesAndClass(data_train2)

# Validation set
data_val2 = csvToList("val_2.csv")
X_val1, y_val1 = getFeaturesAndClass(data_val2)

# Base DT
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train1, y_train1)
y_val_pred1 = clf.predict(X_val1)

cm = confusion_matrix(y_val_pred1, y_val1)
# print(cm)

# Test set
data_test2 = csvToList("test_with_label_2.csv")
X_test1, y_test1 = getFeaturesAndClass(data_test2)

y_test_pred1 = clf.predict(X_test1)

cm = confusion_matrix(y_test_pred1, y_test1)
print(cm)

# Metrics
precision(cm)
recall(cm)
f1_measure(cm)