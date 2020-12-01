import matplotlib.pyplot as plt
import numpy as np 
from Tree import Tree
import pandas as pd
import random
import scipy.stats
import pickle
from bagged_trees import bagged_trees, bagged_trees_pred, ccr


# Importing the datasets
datasets = pd.read_csv('Social_Network_Ads.csv')
X = datasets.iloc[:, [2,3]].values
Y = datasets.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.25, random_state = 0)
Y_Train = np.where(Y_Train==0, -1, Y_Train)
print(type(Y_Train))
print(type(X_Train))

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_Train = sc_X.fit_transform(X_Train)

depth = 10
tree = Tree(None,depth,None,None,0)
tree = Tree.make_tree(tree,X_Train,Y_Train,1,1)

ccr = 0
counter = 0
for i in data:
    point = np.array([i])
    ccr += (labels[counter] == Tree.evaluate_point(tree,point))
    counter +=1

print(counter)
print("This is the CCR " , str(ccr/(counter)))