from adaboost_alt import AdaBst_Alt
import numpy as np
import matplotlib.pyplot as plt
from visuals_alt import plotres
from copy import deepcopy as dc
from WeightedTree_opt import WeightedTree
import pandas as pd

# Importing the datasets
datasets = pd.read_csv('linearly_separable_in_one_d.csv')

# # Import secdon datasets
# datasets2 = pd.read_csv('linearly_separable_in_two_d.csv')

# #Import third dataset
# datasets = pd.read_csv('almost_linearly_separable.csv')



X = datasets.iloc[:, [0,1]].values
Y = datasets.iloc[:, 2].values

# # Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.25, random_state = 0)


# X = np.array([[1,1],[1,0.5],[1,0.75],[1,1.2]])
# Y = np.array([1, -1, -1, -1, 1, -1, 1, -1, -1, 1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1])
# Y= np.array([1,1,-1,-1])
#get results from adaboost classifier
Xcls1 = X[Y==1]
Xcls2 = X[Y==-1]
figure, axes = plt.subplots(figsize=(10, 10), dpi=50)
axes.scatter(*Xcls1.T, marker='.', color='royalblue')
axes.scatter(*Xcls2.T, marker='.', c='mediumvioletred')
axes.set_xlabel('x1')
axes.set_ylabel('x2')
axes.set_title('Full Data Set With Labels')
plt.show()
print(X)
print(Y)
# Y = Y*2-1
tmax = 20

#get results from adaboost classifier
res = AdaBst_Alt(X_Train, Y_Train, tmax)
res.adaclassifier()
# file = open('circles_tmax=75', 'wb')
# dump information to that file
# pickle.dump(res, file) 
# close the file
# file.close()

ts = np.arange(1, tmax+1)
ccr = np.zeros(shape=tmax)
for t in range(tmax):
    curres = dc(res)
    curres.st = res.st[:t]
    curres.st_w = res.st_w[:t]
    Ypred = curres.predict(X_Train)
    ccr[t] = sum(Ypred == Y_Train) / Y_Train.size

plt.figure(1)
plt.plot(ts, ccr)
plt.title('Number of Weak Learners in Ensemble Learner (n) vs Training CCR')
plt.xlabel("n")
plt.ylabel("CCR")
print('\nTraining CCR is: ' + str(ccr[-1]) + '\n')

ts = np.arange(1, tmax+1)
ccr = np.zeros(shape=tmax)

for t in range(tmax):
    curres = dc(res)
    curres.st = res.st[:t]
    curres.st_w = res.st_w[:t]
    Ypred = curres.predict(X_Test)
    ccr[t] = sum(Ypred == Y_Test) / Y_Test.size

plt.figure(2)
plt.plot(ts, ccr)
plt.title('Number of Weak Learners in Ensemble Learner (n) vs Testing CCR')
plt.xlabel("n")
plt.ylabel("CCR")

print('\nTesting CCR is: ' + str(ccr[-1]) + '\n')

# plotres(X_Train, Y_Train, DR=res, axes=None, weights=None)​
plt.show()
# showing iterative results of stump vs ensemble decision real
iterlist = list([0, 4,8,12,16])

fig, axes = plt.subplots(figsize=(8, len(iterlist) * 3),
                         nrows=len(iterlist),
                         ncols=2,
                         sharex=True,
                         dpi=100)


_ = fig.suptitle('Decision boundaries by iteration')
iter = 0
for i in iterlist:

    axstump, axboost = axes[iter]

    print('Predicting Ensemble Decision ' + str(i))
    # Plot weak learner
    _ = axstump.set_title(f'Decision Stump t={i + 1}')
    plotres(X=X, Y=Y,
            stump=res.st[i], weights=res.smp_w[i],
            axes=axstump)

    # Plot strong learner
    curres = dc(res)
    curres.st = res.st[:i+1]
    curres.st_w = res.st_w[:i+1]
    _ = axboost.set_title(f'Aggregated Learner Using t={i + 1} Stumps')
    plotres(X=X, Y=Y,
            DR=curres, weights=curres.smp_w[i],
            axes=axboost)

    iter += 1

plt.tight_layout()
plt.show()
