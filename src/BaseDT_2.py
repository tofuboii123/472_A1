from sklearn import tree
from sklearn.tree import export_text
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
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
clf = tree.DecisionTreeClassifier(criterion="entropy")
clf = clf.fit(X_train2, y_train2)
y_val_pred2 = clf.predict(X_val2)

cm = confusion_matrix(y_val_pred2, y_val2)
# print(cm)

# Test set
data_test2 = csvToList("test_with_label_2.csv")
X_test2, y_test2 = getFeaturesAndClass(data_test2)
plotClassInstances(y_test2, 2, "Plotting of the actual test results of BaseDT_2")

y_test_pred2 = clf.predict(X_test2)
plotClassInstances(y_test_pred2, 2, "Plotting of the predicted results of BaseDT_2")

createCSV("Base-DT-DS2", y_test_pred2)

createCSV("Base-DT-DS2", y_test_pred2)

cm = confusion_matrix(y_test_pred2, y_test2)
print(cm)

# Metrics
precision, recall, f1, accuracy, f1_macro, f1_weight = getMetrics(y_test2, y_test_pred2)
writeMetrics("Base-DT-DS2-Metrics", precision, recall, f1, accuracy, f1_macro, f1_weight)

# Save confusion matrix as csv
np.savetxt("output\Base-DT-DS2-Confusion_Matrix.csv", cm, delimiter=",", fmt='%s')