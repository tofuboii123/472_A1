from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
import numpy as np
from utility import *

# Getting the Training Data Set 
data_train2 = csvToList("train_2.csv")
X_train2, y_train2 = getFeaturesAndClass(data_train2)

# Getting the Validation Set 
data_val2 = csvToList("val_2.csv")
X_val2, y_val2 = getFeaturesAndClass(data_val2)

# Setting the number of iteration with activation logistic and solver gradient
clf = MLPClassifier(max_iter=3000, activation='logistic', solver='sgd')
clf.fit(X_train2, y_train2)

# Get Validation Set 
data_test2 = csvToList("test_with_label_2.csv")

# Separate features from classes
X_test2, y_test2 = getFeaturesAndClass(data_test2)
plotClassInstances(y_test2, 2, "Plotting of the actual test results for BaseMLP_2")

# Prediction 
y_test_pred2 = clf.predict(X_test2)
plotClassInstances(y_test_pred2, 2, "Plotting of the predicted results for BaseMLP_2")

createCSV("Base-MLP-DS2", y_test_pred2)

cm = confusion_matrix(y_test_pred2, y_test2)
print(cm)

# Metrics
precision, recall, f1, accuracy, f1_macro, f1_weight = getMetrics(y_test2, y_test_pred2)
writeMetrics("Base-MLP-DS2-Metrics", precision, recall, f1, accuracy, f1_macro, f1_weight)

print(clf.score(X_test2, y_test2))