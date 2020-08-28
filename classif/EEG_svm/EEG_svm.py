# EEG SVM classification performance analysis
#
# The "ranked_cross_features400.npy" is provided by cross_term_ofr algorithm.
#
# Author: ROBALDO Axel for PI-Psy Institute
# Date: august 2020
#


import numpy as np
import pandas as pd

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, precision_score, accuracy_score



X = np.load("ranked_cross_features400.npy")
y = np.concatenate((np.ones(int(X.shape[0]/2)).T,np.zeros(int(X.shape[0]/2)).T))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)



# LINEAR kernel model
clf = svm.SVC(kernel='linear') # Linear Kernel
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)
print(classification_report(y_test, y_pred))
print('Accuracy linear: ' + str(accuracy_score(y_test, y_pred)))
