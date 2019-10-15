# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 19:35:47 2019

@author: Marek
"""

from collab_model import collab_model
import pandas as pd
import numpy as np

y = np.load('Y.npy')
R = np.load('R.npy')
train = pd.read_csv(r'data/train.csv')

collab = collab_model(learning_rate=0.0003, lamb=0, n_iter=350, n_features=500)
collab.fit(y, R)

predictions=[]
for i in range(len(train)):
	joke = train.joke_id.iloc[i]-1
	user = train.user_id.iloc[i]-1
	pred = float(collab.predict(joke,user))
	predictions.append(pred)

data = pd.DataFrame()
data['Rating'] = train.Rating
del(train)
data['pred_collab'] = predictions

#assign best/worst rating to those which are out of rating range
data.loc[data.pred_collab>10] = 10
data.loc[data.pred_collab<-10] = -10

data['diff'] = data.Rating - data.pred_collab

print('Mean absolute err (y_true - y_pred) is: ' + str(abs(data['diff']).mean()))