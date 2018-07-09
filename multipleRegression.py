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

dataset = pd.read_csv('CCMCInput.csv')


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
plt.xlabel("extra_hours")
plt.ylabel("grades")
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
plt.xlabel("extra_hours")
plt.ylabel("grades")
plt.show()

print("Ploynomial Linear Regression Results:")
# The coefficients (m, b) of y = mx + b
print('Coefficients (X, X^2): \n', regr.coef_)
print('Intercept (b): \n', regr.intercept_)

print("Mean residual sum of squares = %.2f"
      % np.mean((regr.predict(X_transform) - CV) ** 2))
print('R2 = %.2f' % regr.score(X_transform,CV))



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
plt.xlabel("extra_hours")
plt.ylabel("grades")
plt.show()

print("Ploynomial Linear Regression Results:")
# The coefficients (m, b) of y = mx + b
print('Coefficients (X, X^2): \n', regr.coef_)
print('Intercept (b): \n', regr.intercept_)

print("Mean residual sum of squares = %.2f"
      % np.mean((regr.predict(X_transform) - CV) ** 2))
print('R2 = %.2f' % regr.score(X_transform,CV))
