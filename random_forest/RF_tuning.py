# Bagged Trees Classifier Functions
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import scipy.stats
import pickle
from RForest import random_forest, random_forest_pred
from Tree_opt import Tree

X_Train = np.load('data/cvd_data/X_train_cvd.npy')
X_Test = np.load('data/cvd_data/X_test_cvd.npy')
Y_Train = np.load('data/cvd_data/y_train_cvd.npy')
Y_Test = np.load('data/cvd_data/y_test_cvd.npy')


Y_Train = 2*Y_Train - 1
print(X_Train)
print(Y_Train)

# # Random Forest Learning and Predictions
depth = 10
sub_features = 'log2' #'sqrt' and int options too
num_trees = 1
bootstrap_ratio = .3
random_forest(X_Train, Y_Train, bootstrap_ratio, sub_features, depth, num_trees)

# Y_Pred = random_forest_pred(X_Train)
# Y_Pred_test = random_forest_pred(X_Test)