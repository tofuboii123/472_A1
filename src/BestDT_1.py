from sklearn import tree
from sklearn.tree import export_text
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
import numpy as np
from metrics import *
from utility import *

# Training set
data_train1 = csvToList("train_1.csv")
X_train1, y_train1 = getFeaturesAndClass(data_train1)

# Validation set
data_val1 = csvToList("val_1.csv")
X_val1, y_val1 = getFeaturesAndClass(data_val1)

dt = tree.DecisionTreeClassifier()

#Hyper-parameter space
parameter_space = {
    'criterion': ['gini','entropy'],
    'max_depth': [10, None],
    'min_samples_split': [2,3,4,5],
    'min_impurity_decrease' : [0.0, 1.0, 2.0, 3.0, 4.0],
    'class_weight': ['balanced', None]
}

#Grid Search
clf = GridSearchCV(dt, parameter_space, n_jobs = -1, cv = 3)
clf.fit(X_train1, y_train1)

#Best parameter set
print('Best parameters found for DT_1:\n', clf.best_params_)

# Validation set
data_test1 = csvToList("test_with_label_1.csv")

# Separate features from classes
X_test1, y_test1 = getFeaturesAndClass(data_test1)

y_test_pred1 = clf.predict(X_test1)

cm = confusion_matrix(y_test_pred1, y_test1)
print(cm)

# Metrics
getMetrics(y_test1, y_test_pred1)