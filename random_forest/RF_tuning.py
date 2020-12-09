# Bagged Trees Classifier Functions
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import scipy.stats
import pickle
from RForest import random_forest, random_forest_pred
from Tree_opt import Tree



# X_Train = np.load('data/cvd_data/X_train_cvd.npy')
# X_Test = np.load('data/cvd_data/X_test_cvd.npy')
# Y_Train = np.load('data/cvd_data/y_train_cvd.npy')
# Y_Test = np.load('data/cvd_data/y_test_cvd.npy')


# # Random Forest Learning and Predictions
# Y_Train = 2*Y_Train - 1
# Y_Test = 2*Y_Test - 1

# depth = 10
# sub_features = 1 #'log2' #'sqrt' and int options too
# num_trees = 10
# bootstrap_ratio = 1


# # Training and Testing CCR
# def ccr(X_Train, Y_Train, X_Test, Y_Test, bootstrap_ratio, sub_features, depth, num_trees):
# 	random_forest(X_Train, Y_Train, bootstrap_ratio, sub_features, depth, num_trees)
# 	y_train_pred = random_forest_pred(X_Train)
# 	y_test_pred = random_forest_pred(X_Test)
# 	trainccr = sum(y_train_pred==Y_Train)/Y_Train.size
# 	testccr = sum(y_test_pred==Y_Test)/Y_Test.size
# 	return trainccr, testccr

# for i in range(12)

# 	trainccr, testccr = ccr(X_Train, Y_Train, X_Test, Y_Test, bootstrap_ratio, i, depth, num_trees)
# print(trainccr)
# print(testccr)

