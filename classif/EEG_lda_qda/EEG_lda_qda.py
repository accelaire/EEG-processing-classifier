# EEG LDA dimensionality reduction
#
# Author: ROBALDO Axel for PI-Psy Institute
# Date: august 2020
#


import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, precision_score, accuracy_score



X = np.load("ranked_cross_features400.npy")
y = np.concatenate((np.ones(int(X.shape[0]/2)).T,np.zeros(int(X.shape[0]/2)).T))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



#    QDA model ------------------------------------------------------------------------------------

estimator_3 = QDA()
parameters_3 = {
    'reg_param': (0.00001, 0.0001, 0.001,0.01, 0.1), 
    'store_covariance': (True, False),
    'tol': (0.0001, 0.001,0.01, 0.1), 
                   }
# with GridSearch
grid_search_qda = GridSearchCV(
    estimator=estimator_3,
    param_grid=parameters_3,
    scoring = 'accuracy',
    n_jobs = -1,
    cv = 5
)
qda =grid_search_qda.fit(X_train, y_train)
y_pred =qda.predict(X_test)


cm = confusion_matrix(y_test, y_pred)
print(cm)
print(classification_report(y_test, y_pred))
print('Accuracy: ' + str(accuracy_score(y_test, y_pred)))



#   LDA model -------------------------------------------------------------------------------------

lda = LDA()
model_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)

classifier = RandomForestClassifier(max_depth=2, random_state=1)
classifier.fit(model_lda, y_train)
y_pred = classifier.predict(X_test_lda)

cm = confusion_matrix(y_test, y_pred)
print(cm)
print(classification_report(y_test, y_pred))
print('Accuracy: ' + str(accuracy_score(y_test, y_pred)))