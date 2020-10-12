from sklearn import tree
from sklearn.tree import export_text
from sklearn.metrics import confusion_matrix
import numpy as np
from metrics import *
from utility import *

# Training set
data_train1 = csvToList("train_1.csv")
X_train1, y_train1 = getFeaturesAndClass(data_train1)

# Validation set
data_val1 = csvToList("val_1.csv")
X_val1, y_val1 = getFeaturesAndClass(data_val1)

# Base DT
clf = tree.DecisionTreeClassifier(criterion="entropy")
clf = clf.fit(X_train1, y_train1)
y_val_pred1 = clf.predict(X_val1)

cm = confusion_matrix(y_val_pred1, y_val1)
# print(cm)

# Test set
data_test1 = csvToList("test_with_label_1.csv")
X_test1, y_test1 = getFeaturesAndClass(data_test1)

y_test_pred1 = clf.predict(X_test1)

cm = confusion_matrix(y_test_pred1, y_test1)
print(cm)

# Metrics
precision(cm)
recall(cm)
f1_measure(cm)