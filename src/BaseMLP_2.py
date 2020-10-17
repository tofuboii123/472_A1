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

# Prediction 
y_test2_pred2 = clf.predict(X_test2)

cm = confusion_matrix(y_test2_pred2, y_test2)
print(cm)

# Metrics
getMetrics(y_test2, y_test2_pred2)

print(clf.score(X_test2, y_test2))