from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
import numpy as np
from utility import *

# Getting the Training Data Set 
data_train1 = csvToList("train_1.csv")
X_train1, y_train1 = getFeaturesAndClass(data_train1)

# Getting the Validation Set 
data_val1 = csvToList("val_1.csv")
X_val1, y_val1 = getFeaturesAndClass(data_val1)

# Setting the number of iteration with activation logistic and solve gradient
clf = MLPClassifier(max_iter=3500, activation='logistic', solver='sgd')
clf.fit(X_train1, y_train1)

# Validation Set 
data_test1 = csvToList("test_with_label_1.csv")

# Separate features from classes
X_test1, y_test1 = getFeaturesAndClass(data_test1)

# Prediction 
y_test1_pred1 = clf.predict(X_test1)

cm = confusion_matrix(y_test1_pred1, y_test1)
print(cm)

# Metrics
getMetrics(y_test1, y_test1_pred1)

print(clf.score(X_test1, y_test1))