# Bagged Trees Classifier Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import scipy.stats
import pickle
from RForest import random_forest, random_forest_pred
from Tree_opt import Tree


# Importing the datasets
datasets = pd.read_csv('Social_Network_Ads.csv')
X = datasets.iloc[:, [2,3]].values
Y = datasets.iloc[:, 4].values
Y = np.where(Y==0, -1, Y)

# # Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_Train = sc_X.fit_transform(X_Train)
X_Test = sc_X.transform(X_Test)

# Random Forest Learning and Predictions
depth = 10
sub_features = 'log2' #'sqrt' and int options too
num_trees = 10
bootstrap_ratio = .3
random_forest(X_Train, Y_Train, bootstrap_ratio, sub_features, depth, num_trees)
Y_Pred = random_forest_pred(X_Train)
Y_Pred_test = random_forest_pred(X_Test)

# Training and Testing CCR
trainccr = sum(Y_Pred==Y_Train)/Y_Train.size
testccr = sum(Y_Pred_test==Y_Test)/Y_Test.size
print("Training CCR is: " + str(trainccr))
print("Testing CCR is: " + str(testccr))

# Importing the datasets for Circle Noise 
datasets = pd.read_csv('data/circle_noise.csv')
X = datasets.iloc[:, [0,1]].values
Y = datasets.iloc[:, 2].values


X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.25, random_state = 0)
print(X)
# Random Forest Learning and Predictions for Circle Noise
depth = 10
sub_features = 'log2' #'sqrt' and int options too
num_trees = 10
bootstrap_ratio = .3
random_forest(X_Train, Y_Train, bootstrap_ratio, sub_features, depth, num_trees)
Y_Pred = random_forest_pred(X_Train)
Y_Pred_test = random_forest_pred(X_Test)

# Training and Testing CCR for Circle Noise
trainccr = sum(Y_Pred==Y_Train)/Y_Train.size
testccr = sum(Y_Pred_test==Y_Test)/Y_Test.size
print("Training CCR is: " + str(trainccr))
print("Testing CCR is: " + str(testccr))

# from sklearn.ensemble import RandomForestClassifier
# clf = RandomForestClassifier(n_estimators=10, max_depth=10, random_state=0)
# clf.fit(X_Train, Y_Train)
# Y_Pred = clf.predict(X_Train)
# Y_Pred_test = clf.predict(X_Test)

# print(sum(Y_Pred == Y_Train)/Y_Train.size)
# print(sum(Y_Pred_test == Y_Test)/Y_Test.size)
