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
data_test2 = csvToList("test_with_label_1.csv")

# Separate features from classes
X_test2, y_test2 = getFeaturesAndClass(data_test2)

y_test_pred2 = clf.predict(X_test2)

cm = confusion_matrix(y_test_pred2, y_test2)
print(cm)

# Metrics
getMetrics(y_test2, y_test_pred2)


