# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 23:19:58 2020

@author: bkorzen
"""

import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error

df = pd.read_csv('obesity_cleaned.csv')
df.columns

# choose relevant columns
df_model = df[['Gender', 'CAEC', 'TUE', 'CALC', 'MTRANS', 'family_history_with_overweight_binary',
               'FAVC_binary', 'SMOKE_binary', 'SCC_binary', 'age_int', 'FCVC_cat', 'NCP_cat', 'CH2O_cat',
               'FAF_cat', 'TUE_cat', 'bmi']]


# get dummy data
df_dummy = pd.get_dummies(df_model)
df_dummy.columns

# train test split
X = df_dummy.drop('bmi', axis=1)
y = df_dummy['bmi'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# multiple linear regression
X_sm = sm.add_constant(X)
model = sm.OLS(y, X_sm)
model.fit().summary()

# linear regression
lm = LinearRegression()
lm.fit(X_train, y_train)

np.mean(cross_val_score(lm, X_train, y_train, scoring='neg_mean_absolute_error', cv=5))


# lasso regression
lm_lasso = Lasso(alpha=0)
lm_lasso.fit(X_train, y_train)
np.mean(cross_val_score(lm_lasso, X_train, y_train, scoring='neg_mean_absolute_error', cv=5))

alpha = []
errors = []

for i in range(1,100):
    alpha.append(i/1000-0.001)
    lm_lasso = Lasso(alpha=(i/1000))
    errors.append(np.mean(cross_val_score(lm_lasso, X_train, y_train, scoring='neg_mean_absolute_error', cv= 5)))
    
plt.plot(alpha,errors)

err = tuple(zip(alpha,errors))
df_err = pd.DataFrame(err, columns = ['alpha','errors'])
df_err[df_err.errors == max(df_err.errors)]

# random forest
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
np.mean(cross_val_score(rf, X_train, y_train, scoring = 'neg_mean_absolute_error', cv=5))

# GridsearchCV
params = {'n_estimators':range(10,200,10), 'criterion':('mse','mae'), 'max_features':('auto','sqrt','log2')}
gs = GridSearchCV(estimator=rf, param_grid=params, scoring='neg_mean_absolute_error',cv=5)
gs.fit(X_train,y_train)

gs.best_score_
gs.best_estimator_

# test models
tpred_lm = lm.predict(X_test)
tpred_lm_lasso = lm_lasso.predict(X_test)
tpred_rf = gs.best_estimator_.predict(X_test)

mean_absolute_error(y_test, tpred_lm)
mean_absolute_error(y_test, tpred_lm_lasso)   
mean_absolute_error(y_test, tpred_rf)










