from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
from utility import *

# Training set
data_train1 = csvToList("train_1.csv")
X_train1, y_train1 = getFeaturesAndClass(data_train1)

# Validation set
data_val1 = csvToList("val_1.csv")
X_val1, y_val1 = getFeaturesAndClass(data_val1)

mlp = MLPClassifier(max_iter=5000)

# Different params
parameter_space = {
    'hidden_layer_sizes' : [(30, 50), (10, 10, 10)],
    'activation' : ['logistic', 'tanh', 'relu', 'identity'],
    'solver': ['sgd', 'adam']
}

clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)
clf.fit(X_train1, y_train1)

print("Best parameters found:\n", clf.best_params_)

# Validation set
data_test1 = csvToList("test_with_label_1.csv")

# Separate features from classes
X_test1, y_test1 = getFeaturesAndClass(data_test1)

y_test_pred1 = clf.predict(X_test1)
plotClassInstances(y_test_pred1, 1, "Predicted results for BestMLP_1")

createCSV("Best-MLP-DS1", y_test_pred1)

createCSV("Best-MLP-DS1", y_test_pred1)

cm = confusion_matrix(y_test_pred1, y_test1)
print(cm)

# Metrics
precision, recall, f1, accuracy, f1_macro, f1_weight = getMetrics(y_test1, y_test_pred1)
writeMetrics("Best-MLP-DS1-Metrics", precision, recall, f1, accuracy, f1_macro, f1_weight)

