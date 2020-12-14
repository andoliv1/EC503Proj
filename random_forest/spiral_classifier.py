# Bagged Trees Classifier Functions
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import scipy.stats
import pickle
from RForest import random_forest, random_forest_pred
from Tree_opt import Tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Importing the datasets for spirals 
datasets = pd.read_csv('data/spirals.csv')
X = datasets.iloc[:, [0,1]].values
Y = datasets.iloc[:, 2].values
Y[Y==0] = -1
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.10, random_state = 0)

# Using Hand-Built model and ideal hyperparameters, find CCR:
def ccr(X_Train, Y_Train, X_Test, Y_Test, bootstrap_ratio, sub_features, depth, num_trees):
	random_forest(X_Train, Y_Train, bootstrap_ratio, sub_features, depth, num_trees)
	y_train_pred = random_forest_pred(X_Train)
	y_test_pred = random_forest_pred(X_Test)
	trainccr = sum(y_train_pred==Y_Train)/Y_Train.size
	testccr = sum(y_test_pred==Y_Test)/Y_Test.size
	return trainccr, testccr

# Num_Trees v. CCR (using ideal bootstrap, sub_features, and depth Hyperparameters)
def treetest(X_Train, Y_Train, X_Test, Y_Test, tuning_list, reps):
	print("\nIterations:")
	# Testing Random Trees (from scratch)
	depth = 10
	sub_features = 1
	bootstrap_ratio = 1
	avg_trainccr = np.ravel(np.zeros(shape = (1,len(tuning_list))))
	avg_testccr = np.ravel(np.zeros(shape = (1,len(tuning_list))))
	idx = 0
	for i in tuning_list:
		print(i)
		num_trainccr = 0
		num_testccr = 0
		for j in range(reps):
			[trainccr, testccr] = ccr(X_Train, Y_Train, X_Test, Y_Test, bootstrap_ratio, sub_features, depth, i)
			num_trainccr = num_trainccr + trainccr
			num_testccr = num_testccr + testccr
		avg_trainccr[idx] = num_trainccr/reps
		avg_testccr[idx] = num_testccr/reps
		idx+=1
	return avg_trainccr, avg_testccr

# TESTING of Hyperparameters occurs below. 
# IDEAL HYPERPARAMETERS have already been identified.

# Creation of Ideal Model (with all ideal Hyperparameters)
depth = 10
bootstrap_ratio = 1
sub_features = 1
num_trees = 50
[trainccr, testccr] =ccr(X_Train, Y_Train, X_Test, Y_Test, bootstrap_ratio, sub_features, depth, num_trees)
print("Training CCR on RF with ideal Hyperparameters: " + str(trainccr))
print("Testing CCR on RF with ideal Hyperparameters: " + str(testccr))

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_Set, Y_Set = X_Train, Y_Train
X1, X2 = np.meshgrid(np.arange(start = X_Set[:, 0].min() - 1, stop = X_Set[:, 0].max() + 1, step = .2),
                     np.arange(start = X_Set[:, 1].min() - 1, stop = X_Set[:, 1].max() + 1, step = .2))
pred = random_forest_pred((np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape)
plt.contourf(X1, X2, pred, alpha = 0.1, cmap = ListedColormap(('royalblue', 'mediumvioletred')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_Set)):
    plt.scatter(X_Set[Y_Set == j, 0], X_Set[Y_Set == j, 1], s = 20,
                c = ListedColormap(('royalblue', 'mediumvioletred'))(i), label = j)
plt.title('Random Forest Decision Boundary (Training set)')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_Set, Y_Set = X_Train, Y_Train
X1, X2 = np.meshgrid(np.arange(start = X_Set[:, 0].min() - 1, stop = X_Set[:, 0].max() + 1, step = .2),
                     np.arange(start = X_Set[:, 1].min() - 1, stop = X_Set[:, 1].max() + 1, step = .2))
pred = random_forest_pred((np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape)
plt.contourf(X1, X2, pred, alpha = 0.1, cmap = ListedColormap(('royalblue', 'mediumvioletred')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
X_Set, Y_Set = X_Test, Y_Test
for i, j in enumerate(np.unique(Y_Set)):
    plt.scatter(X_Set[Y_Set == j, 0], X_Set[Y_Set == j, 1], s = 20,
                c = ListedColormap(('royalblue', 'mediumvioletred'))(i), label = j)
plt.title('Random Forest Decision Boundary (Testing set)')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.show()

# UNCOMMENT IF NUMBER OF TREES v. CCR IS DESIRED (TAKES A LONG TIME)
# tuning_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
# [avg_trainccr, avg_testccr] = treetest(X_Train, Y_Train, X_Test, Y_Test, tuning_list, 5)
# print(avg_trainccr)
# print(avg_testccr)
# # Num_Trees v. CCR plot
# plt.plot(tuning_list, avg_trainccr)
# plt.title('Average Training CCR v. Number of Trees')
# plt.xlabel('Number of Trees')
# plt.ylabel('Average Training CCR')
# plt.show()
# plt.plot(tuning_list, avg_testccr)
# plt.title('Average Testing CCR v. Number of Trees')
# plt.xlabel('Number of Trees')
# plt.ylabel('Average Testing CCR')
# plt.show()
