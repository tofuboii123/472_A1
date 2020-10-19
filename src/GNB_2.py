from sklearn.naive_bayes import GaussianNB
import numpy as np
from metrics import *
from utility import *

# Training data set
data_train2 = csvToList("train_2.csv")
# Separate features from classes
X_train2, y_train2 = getFeaturesAndClass(data_train2)

# Validation set
data_val2 = csvToList("val_2.csv")

# Separate features from classes
X_val2, y_val2 = getFeaturesAndClass(data_val2)

# Apply GNB model
gnb = GaussianNB()
y_val_pred2 = gnb.fit(X_train2, y_train2).predict(X_val2)

cm = confusion_matrix(y_val2, y_val_pred2)
# print(cm)
print((y_val2 != y_val_pred2).sum())

# Validation set
data_test2 = csvToList("test_with_label_2.csv")

# Separate features from classes
X_test2, y_test2 = getFeaturesAndClass(data_test2)
plotClassInstances(y_test2, 2, "Plotting of the actual results for GNB_2")


# Predict test values
y_test_pred2 = gnb.fit(X_train2, y_train2).predict(X_test2)
plotClassInstances(y_test_pred2, 2, "Plotting of the predicted results for GNB_2")


# Create csv
createCSV("GNB-DS2", y_test_pred2)

cm_test = confusion_matrix(y_test_pred2, y_test2)
print((y_test2 != y_test_pred2).sum())
print(cm_test)

# Metrics
precision, recall, f1, accuracy, f1_macro, f1_weight = getMetrics(y_test2, y_test_pred2)
writeMetrics("GNB-DS2-Metrics", precision, recall, f1, accuracy, f1_macro, f1_weight)

# Save confusion matrix as csv
np.savetxt("output\GNB-DS2-Confusion_Matrix.csv", cm, delimiter=",", fmt='%s')