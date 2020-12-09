# Bagged Trees Classifier Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import scipy.stats
import pickle
from RForest import random_forest, random_forest_pred
from sklearn.datasets import make_gaussian_quantiles
from Tree_opt import Tree

# setting figure
figure, axes = plt.subplots(figsize=(5, 5), dpi=100)

#initialize sample dataset
mu = 0
sigma = 1
n = 1000

X, Y = make_gaussian_quantiles(n_samples=n, n_classes=2, n_features=2)

Y = Y*2-1

# # Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

# Random Forest Learning and Predictions
depth =4
sub_features = 'log2' #'sqrt' and int options too
num_trees = 15
bootstrap_ratio = .3
random_forest(X_Train, Y_Train, bootstrap_ratio, sub_features, depth, num_trees)
Y_Pred = random_forest_pred(X_Train)
Y_Pred_test = random_forest_pred(X_Test)

# Training and Testing CCR
trainccr = sum(Y_Pred==Y_Train)/Y_Train.size
testccr = sum(Y_Pred_test==Y_Test)/Y_Test.size
print("Training CCR is: " + str(trainccr))
print("Testing CCR is: " + str(testccr))

x1, x2 = np.meshgrid(np.arange(-4, 4,0.01),
                             np.arange(-4, 4, 0.01))

Xcls1 = X[Y == 1]
Xcls2 = X[Y == -1]

axes.scatter(*Xcls1.T, marker='.', color='mediumvioletred')
axes.scatter(*Xcls2.T, marker='.', c='royalblue')

print("hello")
Ydec = random_forest_pred(np.c_[x1.ravel(), x2.ravel()])
print("done")
# print("This is theresult of predict")
Ydec = Ydec.reshape(x1.shape)

if sum(np.unique(Ydec)) == 0:
    class_regions = ['royalblue', 'mediumvioletred']
elif sum(np.unique(Ydec)) == 1:
    class_regions = ['royalblue']
else:
    class_regions = ['mediumvioletred']


axes.contourf(x1, x2, Ydec, colors=class_regions, alpha=0.1)
axes.set_xlabel('x_1')
axes.set_ylabel('x_2')
axes.set_title('Plot of Random forests on almosts linearly separable dataset')
plt.show()


