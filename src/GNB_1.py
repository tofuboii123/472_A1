from sklearn.naive_bayes import GaussianNB
import numpy as np
from metrics import *
from utility import *

# Training data set
data_train1 = csvToList("train_1.csv")

# Separate features from classes
X_train1, y_train1 = getFeaturesAndClass(data_train1)

# Validation set
data_val1 = csvToList("val_1.csv")

# Separate features from classes
X_val1, y_val1 = getFeaturesAndClass(data_val1)

# Apply GNB model
gnb = GaussianNB()
y_val_pred1 = gnb.fit(X_train1, y_train1).predict(X_val1)

cm = confusion_matrix(y_val_pred1, y_val1)
# print(cm)
print((y_val1 != y_val_pred1).sum())

# Validation set
data_test1 = csvToList("test_with_label_1.csv")

# Separate features from classes
X_test1, y_test1 = getFeaturesAndClass(data_test1)

# Predict test values
y_test_pred1 = gnb.fit(X_train1, y_train1).predict(X_test1)
cm_test = confusion_matrix(y_test_pred1, y_test1)
print(cm_test)
print((y_test1 != y_test_pred1).sum())

# Metrics
getMetrics(y_test1, y_test_pred1)