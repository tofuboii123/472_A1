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
data_train2 = csvToList("train_2.csv")
X_train2, y_train2 = getFeaturesAndClass(data_train2)

# Validation set
data_val2 = csvToList("val_2.csv")
X_val2, y_val2 = getFeaturesAndClass(data_val2)

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
clf.fit(X_train2, y_train2)

#Best parameter set
print('Best parameters found for DT_2:\n', clf.best_params_)

# Validation set
data_test2 = csvToList("test_with_label_2.csv")

# Separate features from classes
X_test2, y_test2 = getFeaturesAndClass(data_test2)
plotClassInstances(y_test2, 2, "Plotting of the test results for BestDT_2")

y_test_pred2 = clf.predict(X_test2)
plotClassInstances(y_test_pred2, 2, "Plotting of the predicted results for BestDT_2")

createCSV("Best-DT-DS2", y_test_pred2)


createCSV("Best-DT-DS2", y_test_pred2)


cm = confusion_matrix(y_test_pred2, y_test2)
print(cm)

# Metrics
precision, recall, f1, accuracy, f1_macro, f1_weight = getMetrics(y_test2, y_test_pred2)
writeMetrics("Best-DT-DS2-Metrics", precision, recall, f1, accuracy, f1_macro, f1_weight)

# Save confusion matrix as csv
np.savetxt("output\Best-DT-DS2-Confusion_Matrix.csv", cm, delimiter=",", fmt='%s')