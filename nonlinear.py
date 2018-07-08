# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 21:39:12 2018

@author: JiaNian Cen
"""

import pandas as pd
import matplotlib.pyplot as plt
import math
import pylab as P
import numpy as np

from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import KFold, cross_val_score, cross_val_predict
from sklearn import preprocessing


#change input file directory here
dataset = pd.read_excel('dataset.xlsx')
#if want ot print out dataset
#print(dataset)

#plot a scatter plot
#the data is exponential
plt.figure(1)
plt.scatter(dataset.x, dataset.y, color='blue')
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# fit a line to this data
# reshape data and CV to be a matrix again
CV_no_ln = dataset.y.reshape((len(dataset.y), 1))
CV_ln = (dataset.y.map(math.log)).reshape((len(dataset.y), 1))
data = dataset.x.reshape((len(dataset.x), 1))

# Create linear regression object on the log transformed data
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(data, CV_ln)

# get the predictions on the training data
predicted_results_ln = regr.predict(data)
predicted_results = np.exp(predicted_results_ln)
print(predicted_results)

# show in non-linear domain
plt.figure(2)
plt.scatter(data, predicted_results, color='green', s=75)
plt.scatter(data, CV_no_ln, color='black')
plt.xlabel("x")
plt.ylabel("y")
plt.show()

print("Results with fitting a linear model to log transformed data:")
# The coefficients (m, b) of ln y = mx + t
print('Coefficients (m): \n', regr.coef_)
print('Intercept (b): \n', regr.intercept_)

# The mean square error MSE or the mean residual sum of square of errors should be less
MSE = mean_squared_error(CV_ln,predicted_results_ln)
RMSE = math.sqrt(MSE)
# Explained variance score: 1 is perfect prediction
R2 = r2_score(CV_ln,predicted_results_ln)

print("Mean residual sum of squares =", MSE)
print("RMSE =", RMSE)
print("R2 =", R2)

# to see how the residual errors behave
residual_error = CV_ln - predicted_results_ln
print("Mean of residuals =", np.mean(residual_error))
print("Standard deviation of residuals =", np.std(residual_error))

plt.figure(3)
plt.plot((-10,60),(0,0), 'r--')
plt.scatter(data,residual_error,label='residual error')
plt.title("Residual plot")
plt.xlabel("x")
plt.ylabel("residual error")
plt.show()

plt.figure(4)
plt.hist(residual_error)
plt.title("Distribution of residuals")
plt.xlabel("residual error")
plt.show()

plt.figure(5)
n, bins, patches = plt.hist(residual_error, 10, normed=1,  alpha = 0.5)
y_pdf = P.normpdf(bins, np.mean(residual_error), np.std(residual_error))
l = P.plot(bins, y_pdf, 'k--', linewidth=1.5)
plt.show()

#what is really happening
plt.figure(6)
plt.plot(data, predicted_results_ln, color='green', linewidth=3)
plt.scatter(data, CV_ln, color='black')
plt.xlabel("x")
plt.ylabel("ln y")
plt.show()

# Calculating scores for the model
scores = cross_val_score(regr, data, CV_ln, cv=10)
print(scores)

print("Accuracy: %0.5f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# Estimating output
estimated_results = cross_val_predict(regr, data, CV_ln, cv=5)
#print(estimated_results)