# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 18:07:39 2019

@author: Marek
"""

import numpy as np

def gen_Y(df_train):
    n_users = len(df_train.user_id.unique())
    n_jokes = len(df_train.joke_id.unique())
    
    Y = np.zeros((n_jokes, n_users))
    R = np.zeros((n_jokes, n_users))
    
    # ID's become zero-indexed so we don't waste space, and id's in the data set start from 1
    for i, (idx, row) in enumerate(df_train.iterrows()):
        Y[row.joke_id - 1][row.user_id - 1] = row.Rating
        R[row.joke_id - 1][row.user_id - 1] = 1
        
    return Y, R

def mean_norm(Y):
    row = Y.shape[0]

    u = np.mean(Y, axis=1)
    u = u.reshape(row,1)
    
    Y_norm = np.subtract(Y,u)
    
    return u, Y_norm

def init_par(n_users, n_jokes, n_features):
    X = np.random.randn(n_jokes, n_features)*0.01
    Theta = np.random.randn(n_users, n_features)*0.01
    
    return X, Theta

def cost(X, Theta, y, lam, R):
    c = 0.5* np.sum(np.power((np.dot(X, Theta.T) - y) * R, 2))
    reg_x = lam/2 * np.sum(np.power(X,2))
    reg_theta = lam/2 * np.sum(np.power(Theta,2))
    return c + reg_x + reg_theta


def grad(X, Theta, Y, lamb, R):
    n_users = np.shape(Theta)[0]
    n_jokes = np.shape(X)[0]

    X_grad = np.zeros(X.shape)
    Theta_grad = np.zeros(Theta.shape)
    
    #fights large RAM use, to do this at onece you need 10GB+
    for i in range(n_jokes):
        idx = np.where(R[i,:] == 1)
        Theta_temp = Theta[idx]
        Y_temp = Y[i, idx]
        
        X_grad[i] = np.dot(np.dot(X[i, :], Theta_temp.T) - Y_temp, Theta_temp) + lamb*X[i]
    

    for j in range(n_users):
        idx = np.where(R[:,j] == 1)
        X_temp = X[idx]
        Y_temp = Y[idx, j]
        
        Theta_grad[j] = np.dot(np.dot(X_temp, Theta[j, :].T) - Y_temp, X_temp) + lamb*Theta[j]

    return X_grad, Theta_grad

#if __name__ == "__main__":
#    import pandas as pd
#    
#    data = pd.read_csv(r'data\train.csv')
#    Y,R = gen_Y(data)
#    np.save('Y.npy', Y)
#    np.save('R.npy', R)
