# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 13:40:40 2019

@author: Marek
"""

import utils
import numpy as np

class collab_model():
    
    def __init__(self, learning_rate=0.1, lamb=1, n_iter=1000, n_features=10):
        self.learning_rate = learning_rate
        self.lamb = lamb
        self.n_iter = n_iter
        self.n_features = n_features
    
    
    def fit(self, Y, R):
        n_jokes = Y.shape[0]
        n_users = Y.shape[1]
        X, Theta = utils.init_par(n_users, n_jokes, self.n_features)
        
        for i in range(self.n_iter):
            J = utils.cost(X, Theta, Y, self.lamb, R)
            X_grad, Theta_grad = utils.grad(X, Theta, Y, self.lamb, R)
            
            X = X - self.learning_rate * X_grad
            Theta = Theta - self.learning_rate * Theta_grad
            print('cost: ' + str(J),', n_iter: '+str(i))
            
        self.features = X
        self.coef = Theta
        self.cost = utils.cost(X, Theta, Y, self.lamb, R)
        print('final cost: '+ str(self.cost))
        return
    
    def predict(self, joke, user):
        X = self.features[joke]
        Theta = self.coef[user]

        pred = np.dot(Theta.T, X)
        return pred
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
