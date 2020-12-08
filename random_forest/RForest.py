# Bagged Trees Classifier Functions
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import scipy.stats
import pickle
# Decision Tree Classifier Function
from Tree_opt import Tree
# Importing the datasets
datasets = pd.read_csv('Social_Network_Ads.csv')
X = datasets.iloc[:, [2,3]].values
Y = datasets.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_Train = sc_X.fit_transform(X_Train)
X_Test = sc_X.transform(X_Test)

# Bootstrapping training set (with replacement)
def bootstrap(X_Train, Y_Train, ratio):
    [n, d] = X_Train.shape
    sample = np.zeros(shape=(n,d))
    fsample = np.zeros(shape=(n,1))
    for i in range(round(ratio*n)):
        idx = random.randint(0, n-1)
        sample[i,:] = X_Train[idx,:]
        fsample[i] = Y_Train[idx]
    return sample, fsample

# Ensemble Learning
def random_forest(X_Train, Y_Train, bootstrap_ratio, sub_features, depth, num_trees):
    if sub_features == 'sqrt':
        sub_features = int(np.sqrt(X_Train.shape[1]))
    elif sub_features == 'log2':
        sub_features = int(np.log2(X_Train.shape[1]))
    else:
        sub_features = int(sub_features)
    classifier = Tree(None,depth,None,None,0)
    savename = 'saved_trees.pkl'
    file = open(savename,'wb') 
    pickle.dump(num_trees, file) # store number of trees as first object in pickle
    for i in range(num_trees):
        [X_sam, Y_sam] = bootstrap(X_Train, Y_Train, bootstrap_ratio)
        tree = classifier
        tree = Tree.make_tree(tree,X_Train,Y_Train,1,sub_features)
        pickle.dump(tree, file)
    file.close

# Ensemble Predictions
def random_forest_pred(X_data):
    filename = 'saved_trees.pkl'
    file = open(filename, 'rb') 
    num_trees = pickle.load(file) # load tree number from pickle
    [n, _] = X_data.shape
    y_pred_mat = np.zeros(shape=(n,num_trees)) # instantiate matrix of predictions from all trees
    for i in range(num_trees):
        tree = pickle.load(file)
        y_pred_mat[:,i] = Tree.evaluate_data(tree, X_data)
    [y_pred,_] = scipy.stats.mode(y_pred_mat, axis = 1) # take mode of the matrix to combine tree predictions
    file.close
    return np.ravel(y_pred)

