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
    e = 0.1
    X = np.random.uniform(e,-e,size=(n_jokes, n_features))
    Theta = np.random.uniform(e,-e,size=(n_users, n_features))
    
    return X, Theta

def cost(X, Theta, y, lam, R):
    c = 0.5* np.sum(((np.dot(X, Theta.T) - y) * R)**2)
    reg_x = lam/2 * np.sum(X**2)
    reg_theta = lam/2 * np.sum(Theta**2)
    return c + reg_x + reg_theta


def sgd(X, Theta, Y, lamb, R, init_learning_rate=0.01, max_iter=10):
    n_users = np.shape(Theta)[0]
    n_jokes = np.shape(X)[0]

    Theta_grad = np.zeros(Theta.shape)
    
    #fights large RAM useage, to do this at onece you need 10GB+
    for i in range(n_jokes):
        idx = np.where(R[i,:] == 1)
        Theta_temp = Theta[idx]
        Y_temp = Y[i, idx]
        # Its faster to fit one example better in fewer total learning loops
        for k in range(max_iter): 
            X_grad = np.dot(np.dot(X[i, :], Theta_temp.T) - Y_temp, Theta_temp) + lamb*X[i]
            X[i,:] = X[i,:] - (learning_rate*X_grad)

    for j in range(n_users):
        idx = np.where(R[:,j] == 1)
        X_temp = X[idx]
        Y_temp = Y[idx, j]
        # Its faster to fit one example better in fewer total learning loops
        for k in range(max_iter):
            Theta_grad = np.dot(np.dot(X_temp, Theta[j, :].T) - Y_temp, X_temp) + lamb*Theta[j]
            Theta[j,:] = Theta[j,:] - learning_rate*Theta_grad
    return X, Theta
    
#if __name__ == "__main__":
#    import pandas as pd
#    
#    data = pd.read_csv(r'data\train.csv')
#    Y,R = gen_Y(data)
#    np.save('Y.npy', Y)
#    np.save('R.npy', R)
