from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
from utility import *

# TODO Takes a really long time...

# Training set
data_train2 = csvToList("train_2.csv")
X_train2, y_train2 = getFeaturesAndClass(data_train2)

# Validation set
data_val2 = csvToList("val_2.csv")
X_val2, y_val2 = getFeaturesAndClass(data_val2)

mlp = MLPClassifier(max_iter=5000)

# Different params
parameter_space = {
    'hidden_layer_sizes' : [(30, 50), (10, 10, 10)],
    'activation' : ['logistic', 'tanh', 'relu', 'identity'],
    'solver': ['sgd', 'adam']
}

clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)
clf.fit(X_train2, y_train2)

print("Best parameters found:\n", clf.best_params_)

# Validation set
data_test2 = csvToList("test_with_label_2.csv")

# Separate features from classes
X_test2, y_test2 = getFeaturesAndClass(data_test2)
plotClassInstances(y_test2, 2, "Plotting of the actual test results for BestMLP_2")

y_test_pred2 = clf.predict(X_test2)
plotClassInstances(y_test_pred2, 2, "Plotting of the predicted results for BestMLP_2")

createCSV("Best-MLP-DS2", y_test_pred2)

cm = confusion_matrix(y_test_pred2, y_test2)
print(cm)

# Metrics
precision, recall, f1, accuracy, f1_macro, f1_weight = getMetrics(y_test2, y_test_pred2)
writeMetrics("Best-MLP-DS2-Metrics", precision, recall, f1, accuracy, f1_macro, f1_weight)

# Save confusion matrix as csv
np.savetxt("output\Best-MLP-DS2-Confusion_Matrix.csv", cm, delimiter=",", fmt='%s')