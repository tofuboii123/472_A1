from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
import numpy as np

# Training data set
# data_train1 = np.genfromtxt("Assig1-Dataset/train_1.csv", delimiter=",")
data_train1 = np.genfromtxt("train_1.csv", delimiter=",")

# Separate features from classes
X_train1 = data_train1[:, :-1]
y_train1 = data_train1[:, -1]

# Validation set
data_val1 = np.genfromtxt("val_1.csv", delimiter=",")
data_val1 = np.genfromtxt("val_1.csv", delimiter=",")

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
# data_test1 = np.genfromtxt("Assig1-Dataset/test_with_label_1.csv", delimiter=",")
data_test1 = np.genfromtxt("test_with_label_1.csv", delimiter=",")

# Separate features from classes
X_test1 = data_test1[:, :-1]
y_test1 = data_test1[:, -1]


# Predict test values
y_test_pred1 = gnb.fit(X_train1, y_train1).predict(X_test1)
cm_test = confusion_matrix(y_test_pred1, y_test1)
print(cm_test)
print((y_test1 != y_test_pred1).sum())

def precision(cm_test):
    labels_2 = 0
    sum = 0
    for row in cm_test:
        for i in row:
            sum += i
        print("The precision of class %d is %d/%d: %f"%(labels_2,row[labels_2], sum, (row[labels_2]/sum)))
        sum = 0
        labels_2 = labels_2+1

def recall(cm_test):
    sum = 0
    for label in range(0,len(cm_test)):
        for row in cm_test:
            sum += row[label]
        print("The recall of class %d is %d/%d: %f"%(label,cm_test[label,label], sum, (cm_test[label,label]/sum)))
        sum = 0

def f1_measure(cm_test):
    labels_2 = 0
    sum_recall = 0
    sum_precision = 0
    for row in cm_test:
        for i in row:
            sum_precision += i
        for row_2 in cm_test:
            sum_recall += row_2[labels_2]
        print("The f1-measure of class %d is (2x%fx%f)/(%f+%f): %f"%(labels_2,(row[labels_2]/sum_precision), (cm_test[labels_2,labels_2]/sum_recall), (row[labels_2]/sum_precision), (cm_test[labels_2,labels_2]/sum_recall), 2*(row[labels_2]/sum_precision)*(cm_test[labels_2,labels_2]/sum_recall)/((row[labels_2]/sum_precision)+(cm_test[labels_2,labels_2]/sum_recall))))
        sum_precision = 0
        sum_recall = 0
        labels_2 = labels_2+1

precision(cm_test)
recall(cm_test)
f1_measure(cm_test)

