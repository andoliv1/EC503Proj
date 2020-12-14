# Bagged Trees Classifier Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import scipy.stats
import pickle
from RForest import random_forest, random_forest_pred
from Tree_opt import Tree

# setting figure
figure, axes = plt.subplots(figsize=(5, 5), dpi=100)

# Importing the datasets
datasets = pd.read_csv('data/linearly_separable_in_two_d.csv')
# there are other datasets you can load if you want namely data/linearly _separable_in_two_d, data/almost_linearly_separable 
# if you want to see some other decision boundaries
X = datasets.iloc[:, [0,1]].values
Y = datasets.iloc[:, 2].values

X_Train = X
Y_Train = Y

# Random Forest Learning and Predictions
depth =5
sub_features = 'log2' #'sqrt' and int options too
num_trees = 8
bootstrap_ratio = .3
random_forest(X_Train, Y_Train, bootstrap_ratio, sub_features, depth, num_trees)
Y_Pred = random_forest_pred(X_Train)

# Training and Testing CCR
trainccr = sum(Y_Pred==Y_Train)/Y_Train.size
print("Training CCR is: " + str(trainccr))

x1, x2 = np.meshgrid(np.arange(0, 100,0.5),
                             np.arange(0, 100, 0.5))

Xcls1 = X[Y == 1]
Xcls2 = X[Y == -1]

axes.scatter(*Xcls1.T, marker='o',color='mediumvioletred')
axes.scatter(*Xcls2.T, marker='o', c='royalblue')

print("hello")
Ydec = random_forest_pred(np.c_[x1.ravel(), x2.ravel()])
print("done")
Ydec = Ydec.reshape(x1.shape)

if sum(np.unique(Ydec)) == 0:
    class_regions = ['royalblue', 'mediumvioletred']
elif sum(np.unique(Ydec)) == 1:
    class_regions = ['royalblue']
else:
    class_regions = ['mediumvioletred']


axes.contourf(x1, x2, Ydec, colors=class_regions, alpha=0.3)
axes.set_xlabel('x_1')
axes.set_ylabel('x_2')
axes.set_title('Plot of Random forests on linearly separable in on dimension dataset')
plt.show()
