# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 21:58:30 2018

@author: JiaNian Cen
"""

import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold, cross_val_score

dataset = pd.read_csv('grades_dataset.csv')
print(dataset)

# prepare datasets to be fed in the regression model
#predict attend class given extra hours and grade
CV =  dataset.attend_class.reshape((len(dataset.attend_class), 1))
data = (dataset.ix[:,'extra_hours':'grade'].values).reshape((len(dataset.attend_class), 2))

# Create a KNN object
LogReg = LogisticRegression()

# Train the model using the training sets
LogReg.fit(data, CV)

# the model
print('Coefficients (m): \n', LogReg.coef_)
print('Intercept (b): \n', LogReg.intercept_)


#predict the class for each data point
predicted = LogReg.predict(data)
print("Predictions: \n", np.array([predicted]).T)

# predict the probability/likelihood of the prediction
print("Probability of prediction: \n",LogReg.predict_proba(data))

print("Accuracy score for the model: \n", LogReg.score(data,CV))

print(metrics.confusion_matrix(CV, predicted, labels=["Yes","No"]))


#ctrl 4 to block ctrl 5 to uncomment
# =============================================================================
# # Calculating 5 fold cross validation results
# model = LogisticRegression()
# kf = KFold(len(CV), n_folds=5)
# scores = cross_val_score(model, data, CV, cv=kf)
# print("Accuracy of every fold in 5 fold cross validation: ", abs(scores))
# print("Mean of the 5 fold cross-validation: %0.2f" % abs(scores.mean()))
# 
# print("Does he attend class, if he gets 60 after putting 100 hours of effort: ", LogReg.predict([100,60]),LogReg.predict_proba([100,60]))
# =============================================================================
