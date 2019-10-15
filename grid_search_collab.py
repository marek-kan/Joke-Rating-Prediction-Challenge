# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 19:52:16 2019

@author: Marek
"""

from collab_model import collab_model
import pandas as pd
import numpy as np

#very simple grid search without cross-validation focusing only on the minimal
#cost on the training data

Y = np.load('Y.npy')
R = np.load('R.npy')
fitted_params = pd.DataFrame(columns=['lambda','n_features','cost'])

lamb = [0.1,0.5,1,5,10]
n_features = [250,300,500,800]
fit_lam = []
fit_feat = []
fit_cost = []

for lam in lamb:
    for features in n_features:
        collab = collab_model(learning_rate=0.0003, lamb=lam, n_iter=330, n_features=features)
        print('training model with lambda: '+str(lam),'\n','n_features: ' + str(features))
        collab.fit(Y,R)
        
        cost = collab.cost
        fit_lam.append(lam)
        fit_feat.append(features)
        fit_cost.append(cost)
        
fitted_params['lambda'] = fit_lam
fitted_params['n_features'] = fit_feat
fitted_params['cost'] = fit_cost

fitted_params.sort_values(['cost'],inplace=True)