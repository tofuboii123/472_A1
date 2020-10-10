from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
import numpy as np
from GNB import precision, recall, f1_measure

# Training data set
data_train1 = np.genfromtxt("train_2.csv", delimiter=",")
# Separate features from classes
X_train1 = data_train1[:, :-1]
y_train1 = data_train1[:, -1]

# Validation set
data_val1 = np.genfromtxt("val_2.csv", delimiter=",")

# Separate features from classes
X_val1 = data_val1[:, :-1]
y_val1 = data_val1[:, -1]

# Apply GNB model
gnb = GaussianNB()
y_val_pred1 = gnb.fit(X_train1, y_train1).predict(X_val1)

cm = confusion_matrix(y_val1, y_val_pred1)
# print(cm)
print((y_val1 != y_val_pred1).sum())

# Validation set
data_test1 = np.genfromtxt("test_with_label_2.csv", delimiter=",")

# Separate features from classes
X_test1 = data_test1[:, :-1]
y_test1 = data_test1[:, -1]


# Predict test values
y_test_pred1 = gnb.fit(X_train1, y_train1).predict(X_test1)
cm_test = confusion_matrix(y_test_pred1, y_test1)
print((y_test1 != y_test_pred1).sum())
print(cm_test)

precision(cm_test)
recall(cm_test)
f1_measure(cm_test)