import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import scipy.stats
import pickle
# Decision Tree Classifier Function
from Tree_opt import Tree

# Bootstrapping training set (with replacement)
def bootstrap(X_Train, Y_Train, ratio):
    [n, d] = X_Train.shape # Determining number of data points in dataset
    sample = np.zeros(shape=(round(ratio*n),d))
    fsample = np.zeros(shape=(round(ratio*n),1))
    for i in range(round(ratio*n)):  # For loop that iterates the (i = sample size) times
        idx = random.randint(0, n-1) # Randomize index
        sample[i,:] = X_Train[idx,:] # Save data point 
        fsample[i] = Y_Train[idx]    # Save respective feature
    return sample, fsample

# Ensemble Learning
def random_forest(X_Train, Y_Train, bootstrap_ratio, sub_features, depth, num_trees):
    if sub_features == 'sqrt': # Takes square root of dataset features
        sub_features = int(np.sqrt(X_Train.shape[1]))
    elif sub_features == 'log2': # Takes log2 of dataset features
        sub_features = int(np.log2(X_Train.shape[1]))
    else:
        sub_features = int(sub_features) # Specifies number of features to randomly consider
    classifier = Tree(None,depth,None,None,0) 
    savename = 'saved_trees.pkl' # Save all generated trees into a pickle file
    file = open(savename,'wb') 
    pickle.dump(num_trees, file) # Store number of trees as first object in pickle (info)
    for i in range(num_trees): 
        [X_sam, Y_sam] = bootstrap(X_Train, Y_Train, bootstrap_ratio) # Bootstrap dataset for each tree
        tree = classifier
        tree = Tree.make_tree(tree,X_Train,Y_Train,1,sub_features)
        print("Tree " + str(i))
        pickle.dump(tree, file) # Save tree to pickle file
    file.close

# Ensemble Predictions
def random_forest_pred(X_data):
    filename = 'saved_trees.pkl'
    file = open(filename, 'rb')  # Open same pickle file
    num_trees = pickle.load(file) # Load tree number from pickle (info)
    [n, _] = X_data.shape
    y_pred_mat = np.zeros(shape=(n,num_trees)) # Instantiate matrix of predictions from all trees
    for i in range(num_trees):
        tree = pickle.load(file) # Load singular tree
        y_pred_mat[:,i] = Tree.evaluate_data(tree, X_data) # Save tree prediction into matrix
    [y_pred,_] = scipy.stats.mode(y_pred_mat, axis = 1) # Take mode of the matrix to combine tree predictions
    file.close
    return np.ravel(y_pred)

