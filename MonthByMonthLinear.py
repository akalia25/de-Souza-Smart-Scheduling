# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 00:22:13 2018

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

dataset = pd.read_csv('ccmctest.csv')
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

i=24
for i in range (24,36):
       results= regr.predict(i)
       print(i,results)
       i=i+1


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
