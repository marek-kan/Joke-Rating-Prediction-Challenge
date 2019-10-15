# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 20:24:57 2019

@author: Marek
"""

import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np

#found out that XGBoost cannt correct error made by collaboration filltering model

#na_grid.csv is created csv file with one feature - prediction by collab_model
#and target is difference (y_true - y_pred), y_pred is made by collab_model
#Idea was that XGB will learn prediction error by collab_model
X = np.asarray(pd.read_csv('na_grid.csv')['pred_collab']).reshape(-1,1)
y = pd.read_csv('na_grid.csv')['diff']

params = {'learning_rate':[0.1,0.01,0.001], 'alpha':[0,0.5,2], 'lambda':[0,0.3,2],
          'gamma':[0,0.5], 'max_depth':[4,6], 'min_child_weight':[1,5], 'n_jobs':[-1]}
corr = xgb.XGBRegressor()
grid = GridSearchCV(corr,params,scoring='neg_mean_squared_error',n_jobs=-1,cv=4,verbose=30)
search = grid.fit(X,y)