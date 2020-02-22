# Joke-Rating-Prediction-Challenge
Hackathon at Analytics Vidhya, reachable at https://datahack.analyticsvidhya.com/contest/jester-practice-problem.

I have contested under nickname Rekmark, finished third.

# Problem Statement
The dataset contains anonymous ratings(-10 to 10) provided by a total of 41,000 users. Train file contains 1.1 million ratings and in the test file the user needs to predict the ratings provided by the same set of users on a diffrent set of jokes. The complete text for all 139 jokes is also provided in a separate csv. Given the combination of user and joke, the task is to predict the rating given by that user to the joke in the test set

# Solution
My first try has been based on NLP algorithm "Bag of Words", combined with some synthetic features (like std_user/std_joke, ...) which have had high correlation with the ratings. These steps have resulted into very large dataset. I have had to use Google colab to be able to feed these data to XGBoost model with gradient boosted trees. At the end this model performed poorly (RMSE ~ 5.8). 
(At this time I havn't been familiar with collaboration filtering algorithm).

To address rating prediction I've used Collaborative Filtering algorithm which I have learned from Week 9, Machine Learning course at Coursera taught by Andrew Ng (https://www.coursera.org/learn/machine-learning). When this model hasn't been optimalized I've used XGBoost model to correct systematic errors. Then I optimalized the collab_model params and found out that XGBoost is no longer able to correct errors in the predictions.

# Project description
Folder "data" contains data.rar file where all data sources are.
data_describtion.txt contains information about data.

* utils.py:

  * gen_Y(train_data) - will generate Y and R (matrix which users rated which jokes) matrices.
  
  * mean_norm(Y) - returns Y_normalized, u (matrix of joke means); has been used in some testing to get better results; not used in final                  solution.
  
  * init_par(n_users, n_jokes, n_features) - initialize feature (X) and coef (Theta) matrices.
  
  * cost(X, Theta, y, lam, R) - calculate current cost.
  
  * sgd(X, Theta, Y, lamb, R) - calculate gradients for X and Theta

* collab_model.py:
  contains collab_model class
  
* train_models.py:
  will train collab model and make predictions on training set. Then will print mean absolute error on training set.

* grid_search_collab.py:
  feature which will perform simple grid search with defined params and store results.

* grid_search_corr.py:
  has been used to find optimal params for correction (XGB) model.
  
  
=======================================================================================================================
Resources:
https://www.coursera.org/learn/machine-learning
Tony Tonev [ https://github.com/tonytonev ]
