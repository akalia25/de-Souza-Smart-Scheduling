#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 13:53:53 2018

@author: adityakalia
"""

# =============
# Introduction
# =============
# I've been doing some data mining lately and specially looking into `Gradient
# Boosting Trees <http://en.wikipedia.org/wiki/Gradient_boosting>`_ since it is
# claimed that this is one of the techniques with best performance out of the
# box.  In order to have a better understanding of the technique I've reproduced
# the example of section *10.14.1 California Housing* in the book `The Elements of Statistical Learning <http://www-stat.stanford.edu/~tibs/ElemStatLearn/>`_.
# Each point of this dataset represents the house value of a property with some
# attributes of that house. You can get the data and the description of those
# attributes from `here <http://lib.stat.cmu.edu/modules.php?op=modload&name=Downloads&file=index&req=getit&lid=83>`_.

# I know that the whole exercise here can be easily done with the **R** package
# `gbm <http://cran.r-project.org/web/packages/gbm/index.html>`_ but I wanted to
# do the exercise using Python. Since learning several languages well enough is
# difficult and time consuming I would prefer to stick all my data analysis to
# Python instead doing it in R, even with R being superior on some cases. But
# having only one language for doing all your scripting, systems programming and
# prototyping *PLUS* your data analysis is a good reason for me. Your upfront
# effort of learning the language, setting up your tools and editors, etc. is
# done only once instead of twice. 
# 
# Data Preparation
# -----------------
# The first thing to do is to load the data into a `Pandas <http://pandas.pydata.org/pandas-docs/stable/>`_  dataframe

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

input2 = '/Users/adityakalia/Downloads/coursesv3.csv'
course_codes = pd.read_csv(input2,header = 0)
df = pd.DataFrame(course_codes)
df.drop(['course_id'], axis = 1)
#Figure out why course_id doesnt drop here, super strange

#columnNames = ['HouseVal','MedInc','HouseAge','AveRooms',
#               'AveBedrms','Population','AveOccup','Latitude','Longitud']

#df = pd.read_csv('cadata.txt',skiprows=27, sep='\s+',names=columnNames)

# Now we have to split the datasets into training and validation. The training
# data will be used to generate the trees that will constitute the final
# averaged model.
#hi
import random

# encode string class values as integers
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(df['course_code'])
course_code_2 = label_encoder.transform(df['course_code'])
label_encoder = label_encoder.fit(df['course'])
course_2 = label_encoder.transform(df['course'])

df2 = pd.DataFrame({'course_2': course_2, 'course_code_2':course_code_2})
df['course'] = df2['course_2']
df['course_code'] = df2['course_code_2']
df['course_enrolment']=df['course_enrolment'].fillna(0)

#X = df[df.columns - ['random_demand']]
X = df.drop(['course_enrolment','course_id','course_starts','course_ends','Date_Diff' ], axis = 1)
Y = df['course_enrolment']

rows = random.sample(list(df.index), int(len(df)*.80))
x_train, y_train = X.ix[rows],Y.ix[rows]
x_test,y_test  = X.drop(rows),Y.drop(rows)

# We then fit a Gradient Tree Boosting model to the data using the
# `scikit-learn <http://scikit-learn.org/stable/>`_ package. We will use 500 trees
# with each tree having a depth of 6 levels. In order to get results similar to
# those in the book we also use the `Huber loss function <http://en.wikipedia.org/wiki/Huber_loss_function>`_ .

from sklearn.metrics import mean_squared_error,r2_score
from sklearn.ensemble import GradientBoostingRegressor
params = {'n_estimators': 500, 'max_depth': 6,
        'learning_rate': 0.1, 'loss': 'huber','alpha':0.95}
#clf = GradientBoostingRegressor(**params).fit(x_train, y_train)

clf = GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=500,
                                subsample=1.0, min_samples_split=2, min_samples_leaf=1,
                                min_weight_fraction_leaf=0.0, max_depth=3, init=None,
                                random_state=None, max_features=None, alpha=0.95, verbose=0, 
                                max_leaf_nodes=None, warm_start=False, presort='auto').fit(x_train,y_train)
#clf = GradientBoostingRegressor().fit(x_train, y_train)
# For me, the Mean Squared Error wasn't much informative and used instead the
# :math:`R^2` **coefficient of determination**. This measure is a number
# indicating how well a variable is able to predict the other. Numbers close to
# 0 means poor prediction and numbers close to 1 means perfect prediction. In the
# book, they claim a 0.84 against a 0.86 reported in the paper that created the
# dataset using a highly tuned algorithm. I'm getting a good 0.83 without much
# tunning of the parameters so it's a good out of the box technique.

mse = mean_squared_error(y_test, clf.predict(x_test))
r2 = r2_score(y_test, clf.predict(x_test))
\
print("MSE: %.4f" % mse)
print("R2: %.4f" % r2)

# Let's plot how does it behave the training and testing error

import matplotlib.pyplot as plt

# compute test set deviance
test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

for i, y_pred in enumerate(clf.staged_decision_function(x_test)):
    test_score[i] = clf.loss_(y_test, y_pred)

plt.figure(figsize=(12, 6))
plt.subplot(1, 1, 1)
plt.title('Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, clf.train_score_, 'b-',
                label='Training Set Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
                label='Test Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')

# As you can see in the previous graph, although the train error keeps going
# down as we add more trees to our model, the test error remains more or less
# constant and doesn't incur in overfitting. This is mainly due to the shrinkage
# parameter and one of the good features of this algorithm.


# When doing data mining as important as finding a good model is being able to
# interpret it, because based on that analysis and interpretation preemptive
# actions can be performed. Although base trees are easily interpretable when
# you are adding several of those trees interpretation is more difficult. You
# usually rely on some measures of the predictive power of each feature. Let's
# plot feature importance in predicting the House Value.

feature_importance = clf.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.figure(figsize=(12, 6))
plt.subplot(1, 1, 1)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, X.columns[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()

# Once variable importance has been identified we could try to investigate how
# those variables interact between them. For instance, we can plot the
# dependence of the target variable with another variable has been averaged over
# the values of the other variables not being taken into consideration. Some
# variables present a clear monotonic dependence with the target value, while
# others seem not very related to the target variable even when they ranked high
# in the previous plot. This could be signaling an interaction between variables
# that could be further studied. 

from sklearn.ensemble.partial_dependence import plot_partial_dependence

fig, axs = plot_partial_dependence(clf, x_train, 
                                   features=[3,2,7,6],
                                   feature_names=x_train.columns,
                                   n_cols=2)

fig.show()
#match predict,x_test
# The last step performed was to explore the capabilities of the Python
# libraries when plotting data in a map. Here we are plotting the predicted
# House Value in California using Latitude and Longitude as the axis for
# plotting this data in the map.


