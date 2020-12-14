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

X_Train = np.load('data/cvd_data/X_train_cvd.npy')
X_Test = np.load('data/cvd_data/X_test_cvd.npy')
Y_Train = np.load('data/cvd_data/y_train_cvd.npy')
Y_Test = np.load('data/cvd_data/y_test_cvd.npy')


# Random Forest Learning and Predictions
Y_Train = 2*Y_Train - 1
Y_Test = 2*Y_Test - 1

# SKLEARN WAS USED BECAUSE COMPUTATION TIME IS MUCH FASTER
# TUNING CAN BE DONE MORE EFFICIENTLY WITH SKLEARN
# Calculating Parameters with skLearn Random Forest
def ccrsk(X_Train, Y_Train, X_Test,Y_Test, n_est, max_sam, max_feat, max_dep):
	clf = RandomForestClassifier(n_estimators=n_est, max_samples=max_sam, max_features=max_feat, max_depth=max_dep,  random_state=0)
	clf.fit(X_Train, Y_Train)
	y_train_pred = clf.predict(X_Train)
	y_test_pred = clf.predict(X_Test)
	trainccr = sum(y_train_pred==Y_Train)/Y_Train.size
	testccr = sum(y_test_pred==Y_Test)/Y_Test.size
	return trainccr, testccr

# Tuning Parameters with skLearn Random Forest
def tuning(X_Train, Y_Train, X_Test, Y_Test, tuning_list, reps):
    print("\nTuning Iterations:")
    avg_trainccr = np.ravel(np.zeros(shape = (1,len(tuning_list))))
    avg_testccr = np.ravel(np.zeros(shape = (1,len(tuning_list))))
    idx = 0
    for i in tuning_list:
        print(idx)
        num_trainccr = 0
        num_testccr = 0
        for j in range(reps):
            # DEPENDING ON PLACEMENT OF i IN ARGUMENT BELOW, CAN ADJUST DIFFERENT PARAMETERS, HOLDING OTHER 3 CONSTANT
            [trainccr, testccr] = ccrsk(X_Train, Y_Train, X_Test, Y_Test, i, .8, 5, 5) # CURRENTLY TUNING NUMBER OF TREES
            num_trainccr = num_trainccr + trainccr
            num_testccr = num_testccr + testccr
        avg_trainccr[idx] = num_trainccr/reps
        avg_testccr[idx] = num_testccr/reps
        idx+=1
    return avg_trainccr, avg_testccr

# Tuning different parameters 
tuning_list = [1,2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100] 
[avg_trainccr, avg_testccr] = tuning(X_Train, Y_Train, X_Test, Y_Test, tuning_list, 3)

# Tuning v. CCR plot
plt.plot(tuning_list, avg_trainccr)
plt.title('Average Training CCR v. Number of Trees')
plt.xlabel('Number of Trees')
plt.ylabel('Average Training CCR')
plt.show()
plt.plot(tuning_list, avg_testccr)
plt.title('Average Testing CCR v. Number of Trees')
plt.xlabel('Number of Trees')
plt.ylabel('Average Testing CCR')
plt.show()

# Testing with Ideal Parameters
trainccr, testccr = ccrsk(X_Train, Y_Train, X_Test,Y_Test, 60, .8, 5, 5)
print("Training CCR with Ideal Parameters are: " + str(trainccr))
print("Testing CCR with Ideal Parameters are: " + str(testccr))

