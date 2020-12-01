# Bagged Trees Classifier Functions
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import scipy.stats
import pickle

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

# Decision Trees Definition
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'gini', random_state = 0)

# Ensemble Learning
def bagged_trees(X_Train, Y_Train, num_trees, classifier, savename):
    ratio = 1
    file = open(savename,'wb') 
    # trees = np.zeros(shape=(num_trees), dtype=object)
    pickle.dump(num_trees, file) # store number of trees as first object in pickle
    for i in range(num_trees):
        [X_sam, Y_sam] = bootstrap(X_Train, Y_Train, ratio)
        tree = classifier
        tree.fit(X_sam,Y_sam)
        # trees[i] = tree
        pickle.dump(tree, file)
    file.close

# Ensemble Predictions
def bagged_trees_pred(X_data, filename):
    file = open(filename, 'rb') 
    num_trees = pickle.load(file) # load tree number from pickle
    [n, _] = X_data.shape
    y_pred_mat = np.zeros(shape=(n,num_trees)) # instantiate matrix of predictions from all trees
    for i in range(num_trees):
        tree = pickle.load(file)
        y_pred_mat[:,i] = tree.predict(X_data)
    [y_pred,_] = scipy.stats.mode(y_pred_mat, axis = 1) # take mode of the matrix to combine tree predictions
    file.close
    return np.ravel(y_pred)

filename = 'saved_trees.pkl'
trees = bagged_trees(X_Train, Y_Train, 3, classifier, filename)
y_pred = bagged_trees_pred(X_Train, filename)


# Training and Testing CCR
def ccr(X_Train, Y_Train, X_Test, Y_Test, classifier, num_trees, filename):
    trees = bagged_trees(X_Train, Y_Train, num_trees, classifier, filename)
    y_train_pred = bagged_trees_pred(X_Train, filename)
    y_test_pred = bagged_trees_pred(X_Test, filename)
    trainccr = sum(y_train_pred==Y_Train)/Y_Train.size
    testccr = sum(y_test_pred==Y_Test)/Y_Test.size
    return trainccr, testccr


# Cross-Validation
def crossval(X_Train, Y_Train, X_Test, Y_Test, classifier, filename, num_trees_list, reps):
    print("\nCross Validation Iterations:")
    avg_trainccr = np.ravel(np.zeros(shape = (1,len(num_trees_list))))
    avg_testccr = np.ravel(np.zeros(shape = (1,len(num_trees_list))))
    idx = 0
    for i in num_trees_list:
        print(idx)
        num_trainccr = 0
        num_testccr = 0
        for j in range(reps):
            [trainccr, testccr] = ccr(X_Train, Y_Train, X_Test, Y_Test, classifier, i, filename)
            num_trainccr = num_trainccr + trainccr
            num_testccr = num_testccr + testccr
        avg_trainccr[idx] = num_trainccr/reps
        avg_testccr[idx] = num_testccr/reps
        idx+=1
    return avg_trainccr, avg_testccr


filename = 'saved_trees.pkl'
num_trees_list = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
reps = 50
print("\nCross-Validation Tree Number List:")
print(num_trees_list)
[avg_trainccr, avg_testccr] = crossval(X_Train, Y_Train, X_Test, Y_Test, classifier, filename, num_trees_list, reps)
print("\nAverage Training CCR for model with respective number of trees:")
print(avg_trainccr)
print("\nAverage Testing CCR for model with respective number of trees:")
print(avg_testccr)

plt.plot(num_trees_list, avg_trainccr)
plt.title('Average Training CCR v. Number of Trees')
plt.xlabel('Number of Trees')
plt.ylabel('Average Training CCR')
plt.show()
plt.plot(num_trees_list, avg_testccr)
plt.title('Average Testing CCR v. Number of Trees')
plt.xlabel('Number of Trees')
plt.ylabel('Average Testing CCR')
plt.show()




