# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 21:45:32 2018

@author: JiaNian Cen
"""

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import math
import pylab as P
import numpy as np


from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import *

dataset = pd.read_csv('CCMC001.csv')
#dataset = pd.read_csv('FONPInput.csv')
#dataset = pd.read_csv('APAMInput.csv')
#dataset.course_starts=str(dataset.course_starts-d)

#print(dataset)

#draw data set
plt.figure(1)
plt.title('Course CCMC')
plt.scatter(dataset.Month_Diff,dataset.coure_code_enrolment,  color='blue')
plt.xlabel("Month Diff to 2016-01-01")
plt.ylabel("Enrolment")
plt.show()


#Single Linear Regression
data = dataset.Month_Diff.values.reshape((len(dataset.Month_Diff), 1))
CV = dataset.coure_code_enrolment.values.reshape((len(dataset.coure_code_enrolment), 1))
# Create linear regression object
regr = linear_model.LinearRegression()
# Train the model using the training sets
regr.fit(data, CV)

# get the predictions on the training data
predicted_results = regr.predict(data)

print("Simple Linear Regression Results:")
# The coefficients (m, b) of y = mx + b
print('Coefficients (m1, m2): \n', regr.coef_)
print('Intercept (b): \n', regr.intercept_)

print("Mean residual sum of squares = %.2f"
      % np.mean((regr.predict(data) - CV) ** 2))
print('R2 = %.2f' % regr.score(data,CV))

plt.plot(data, predicted_results, color='green', linewidth=3)
plt.scatter(data, CV, color='black')
plt.xlabel("Month Diff to 2016-01-01")
plt.ylabel("Enrolment")
plt.show()




#multi variable regression
courselength =  dataset.course_length.values.reshape((len(dataset.course_length), 1))
Month_Diff =  dataset.Month_Diff.values.reshape((len(dataset.Month_Diff), 1))

data = data = np.concatenate((courselength, Month_Diff), axis=1)


# Create linear regression object
regr = linear_model.LinearRegression()
# Train the model using the training sets
regr.fit(data, CV)

# get the predictions on the training data
predicted_results = regr.predict(data)

print("Multiple Linear Regression Results:")
# The coefficients (m, b) of y = mx + b
print('Coefficients (m1, m2): \n', regr.coef_)
print('Intercept (b): \n', regr.intercept_)

print("Mean residual sum of squares = %.2f"
      % np.mean((regr.predict(data) - CV) ** 2))
print('R2 = %.2f' % regr.score(data,CV))



#ploynomial OF 2
#https://www.youtube.com/watch?v=EvnpoUTXA0E



# =============================================================================
# data = dataset.Month_Diff
# pl = np.polyfit(data, CV, 1)
# 
# from matplotlib.pyplot import *
# 
# plot(data, CV, 'o')
# plot(data, np.polyval(pl,data),'r-')
# 
# p2 = np.polyfit(data, CV, 2)
# p3 = np.polyfit(data, CV, 10)
# 
# plot(data, np.polyval(p2,data),'b--')
# 
# plot(data, np.polyval(p3,data),'m:')
# =============================================================================

####done plot######

data = dataset.Month_Diff[:,None]
# =============================================================================
# print(data)
# data = np.linspace(-5,5,num=100)[:,None]
# print(data)
# =============================================================================
CV = dataset.coure_code_enrolment
# create a Linear Regressor   
regr = LinearRegression()

# pass the order of your polynomial here  
poly = PolynomialFeatures(degree=2, include_bias=False)

# convert to be used further to linear regression
X_transform = poly.fit_transform(data)

# fit this to Linear Regressor
regr.fit(X_transform,CV) 

# get the predictions on the training data
predicted_results = regr.predict(X_transform)

plt.plot(data, predicted_results, color='red', linewidth=3)
plt.scatter(data, CV, color='blue')
plt.xlabel("Month Diff to 2016-01-01")
plt.ylabel("Enrolment")
plt.show()

print("Ploynomial Linear Regression Results:")
# The coefficients (m, b) of y = mx + b
print('Coefficients (X, X^2): \n', regr.coef_)
print('Intercept (b): \n', regr.intercept_)

print("Mean residual sum of squares = %.2f"
      % np.mean((regr.predict(X_transform) - CV) ** 2))
print('R2 = %.2f' % regr.score(X_transform,CV))
#MSE when 80% of the data is used to obtain a model (training), tested on the rest 20%
i=0
for i in range (0,5):
      X_train, X_test, y_train, y_test = train_test_split(data, CV, test_size=.2)
      X_transform = poly.fit_transform(X_train)
      regr.fit(X_train, y_train)
      print("MSE with 80-20 split iteration %s : %0.2f" % (i , mean_squared_error(y_test,regr.predict(X_test))))
      i=i+1


#run degree with 7
data = dataset.Month_Diff[:,None]
# =============================================================================
# print(data)
# data = np.linspace(-5,5,num=100)[:,None]
# print(data)
# =============================================================================
CV = dataset.coure_code_enrolment
# create a Linear Regressor   
regr = LinearRegression()

# pass the order of your polynomial here  
poly = PolynomialFeatures(degree=7, include_bias=False)

# convert to be used further to linear regression
X_transform = poly.fit_transform(data)

# fit this to Linear Regressor
regr.fit(X_transform,CV) 

# get the predictions on the training data
predicted_results = regr.predict(X_transform)

plt.plot(data, predicted_results, color='red', linewidth=3)
plt.scatter(data, CV, color='blue')
plt.xlabel("Month Diff to 2016-01-01")
plt.ylabel("Enrolment")
plt.show()

print("Ploynomial Linear Regression Results:")
# The coefficients (m, b) of y = mx + b
print('Coefficients (X, X^2...X^7): \n', regr.coef_)
print('Intercept (b): \n', regr.intercept_)

print("Mean residual sum of squares = %.2f"
      % np.mean((regr.predict(X_transform) - CV) ** 2))
print('R2 = %.2f' % regr.score(X_transform,CV))


####Cross validation####
# MSE when all of training data used for testing
#print("Mean residual sum of squares when all the data is used for training: %0.2f" % np.mean((regr.predict(X_transform) - CV) ** 2))

#MSE when 80% of the data is used to obtain a model (training), tested on the rest 20%
i=0
for i in range (0,5):
      X_train, X_test, y_train, y_test = train_test_split(data, CV, test_size=.2)
      X_transform = poly.fit_transform(X_train)
      regr.fit(X_train, y_train)
      print("MSE with 80-20 split iteration %s : %0.2f" % (i , mean_squared_error(y_test,regr.predict(X_test))))
      i=i+1

# =============================================================================
# model = linear_model.LinearRegression()
# # Calculating cross-validated scores for the model
# kf = KFold(len(CV), n_folds=5, shuffle=True, random_state=0)
# scores = cross_val_score(model, data, CV, scoring = 'mean_squared_error', cv=kf)
# print("MSE of every fold with K=5: ", abs(scores))
# print("Mean of 5-fold cross-validated MSE: %0.2f (+/- %0.2f)" % (abs(scores.mean()), scores.std() * 2))
# 
# # Calculating leave one out cross validation scores for the model
# # can use the built in function LeaveOneOut()
# kf = KFold(len(CV), n_folds=10)
# scores = cross_val_score(model, data, CV, scoring = 'mean_squared_error', cv=kf)
# print("MSE of every fold in leave one out cross validation: ", abs(scores))
# print("Mean of 10-fold cross-validated MSE: %0.2f (+/- %0.2f)" % (abs(scores.mean()), scores.std() * 2))
# 
# # Estimating output: what was CV when data row was in training dataset
# estimated_results = cross_val_predict(regr, data, CV, cv=5)
# print(estimated_results)
# https://towardsdatascience.com/machine-learning-with-python-easy-and-robust-method-to-fit-nonlinear-data-19e8a1ddbd49
# =============================================================================
