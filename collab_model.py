# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 13:40:40 2019

@author: Marek
"""

import utils
import numpy as np
import time

class collab_model():
    
    def __init__(self, learning_rate=0.1, lamb=1, n_iter=1000, n_features=10):
        self.learning_rate = learning_rate
        self.lamb = lamb
        self.n_iter = n_iter
        self.n_features = n_features
    

    def fit_sgd(self, Y, R):
        n_jokes = Y.shape[0]
        n_users = Y.shape[1]
        X, Theta = utils.init_par(n_users, n_jokes, self.n_features)
        start = time.time()
        for i in range(self.n_iter):
            
            X, Theta = utils.sgd(X, Theta, Y, self.lamb, R, init_learning_rate=self.learning_rate, max_iter=8)
            J = utils.cost(X, Theta, Y, self.lamb, R)
            print('cost: ' + str(J),', n_iter: '+str(i))
            if J < 200:
                break
        self.features = X
        self.coef = Theta
        self.cost = utils.cost(X, Theta, Y, self.lamb, R)
        end = time.time()
        self.train_time = end-start
        print('final cost: '+ str(self.cost),'\n'
              'train time: '+str(self.train_time))
        return
        
    def predict(self, joke, user):
        X = self.features[joke]
        Theta = self.coef[user]

        pred = np.dot(Theta.T, X)
        return pred

        
