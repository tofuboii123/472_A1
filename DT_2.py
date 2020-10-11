from sklearn import tree
from sklearn.tree import export_text
from sklearn.metrics import confusion_matrix
import numpy as np
from metrics import *
from utility import *

# Training set
data_train2 = csvToList("train_2.csv")
X_train2, y_train2 = getFeaturesAndClass(data_train2)

# Validation set
data_val2 = csvToList("val_2.csv")
X_val2, y_val2 = getFeaturesAndClass(data_val2)

# Base DT
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train2, y_train2)
y_val_pred2 = clf.predict(X_val2)

cm = confusion_matrix(y_val_pred2, y_val2)
# print(cm)

# Test set
data_test2 = csvToList("test_with_label_2.csv")
X_test2, y_test2 = getFeaturesAndClass(data_test2)

y_test_pred2 = clf.predict(X_test2)

cm = confusion_matrix(y_test_pred2, y_test2)
print(cm)

# Metrics
precision(cm)
recall(cm)
f1_measure(cm)