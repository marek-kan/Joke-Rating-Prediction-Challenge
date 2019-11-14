# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 19:35:47 2019

@author: Marek
"""

from collab_model import collab_model
import pandas as pd
import numpy as np
import pickle

y = np.load('Y.npy')
R = np.load('R.npy')
train = pd.read_csv(r'data/train.csv')

collab = collab_model(learning_rate=0, lamb=1e-6, n_iter=350, n_features=2000)
collab.fit_sgd(y, R)

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

filename = 'collab_model_{}'.format(int(collab.cost))
print('Saving model {}'.format(filename))
pickle.dump(collab, open(filename, 'wb'))

