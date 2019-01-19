# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 03:09:56 2018

@author: JiaNian Cen
"""

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

dataset = pd.read_csv('CCMC03mar.csv')
f = open('CCMC03marmulti.csv','w')




data = dataset.Month_Diff[:,None]
# =============================================================================
# print(data)
# data = np.linspace(-5,5,num=100)[:,None]
# print(data)
# =============================================================================
CV = dataset.Enrol_Num
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

k = 0
while k < len(data):
    temp=data[k]
    temp=temp.reshape(-1, 1)
    X_transform3 = poly.fit_transform(temp)
    results1= regr.predict(X_transform3)
    
    f.write(str(temp))
    f.write(',')
    f.write(str(results1))
    f.write('\n')
    
    k += 1




i=data[-1]+12
i=i.reshape(-1, 1)


j=1
while j<3:
       X_transform2 = poly.fit_transform(i)
       results= regr.predict(X_transform2)
       f.write(str(i))
       f.write(',')
       f.write(str(results))
       f.write('\n')
       i=i+12
       j=j+1
       
f.close()
