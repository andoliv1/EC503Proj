from adaboost_alt import AdaBst_Alt
import numpy as np
import matplotlib.pyplot as plt
from visuals_alt import plotres
from copy import deepcopy as dc
from sklearn.datasets import make_gaussian_quantiles


#initialize sample dataset
mu = 0
sigma = 1
n = 100

# X1 = np.random.uniform(-1, 0.5, size=(10, 2))
# X2 = np.random.uniform(, 1, size=(10, 2))
# X = np.zeros((20, 2))
#
# X[0:10, :] = X1
# X[10:20, :] = X2
# print(X.shape)
# Y = np.ones((20,))
# Y[0:6] = 1
# Y[6:11] = -1
# print(Y)
X, Y = make_gaussian_quantiles(n_samples=n, n_classes=2, n_features=2)
# print(X)
# print(Y)
Y = Y*2-1
tmax = 100

# X = np.array([[ 0.91941369, -1.40668594],
#  [ 0.18477678,  0.42003076],
#  [-0.03040438, -0.10652165],
#  [ 0.48855982,  0.06565067],
#  [-0.05002865,  1.44088418],
#  [ 0.33019429, -0.87078287],
#  [-0.58190124,  2.41223044],
#  [-0.65434502,  0.43796255],
#  [-0.37328651,  0.07182572],
#  [-1.2837738 ,  0.40693525],
#  [-0.66555328,  0.55762218],
#  [ 0.7444908 , -0.5746354 ],
#  [ 0.62609619,  0.1515806 ],
#  [-0.40310026, -0.45315985],
#  [ 0.8295343 , -1.01863621],
#  [ 0.07604939,  1.32279631],
#  [ 0.5957569 , -1.43382616],
#  [ 0.35238149, -1.36161462],
#  [-0.73368632, -2.3155115 ],
#  [-0.91909313, -2.07985754]])

# X = np.array([[1,1],[1,0.5],[1,0.75],[1,1.2]])
# Y = np.array([1, -1, -1, -1, 1, -1, 1, -1, -1, 1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1])
# Y= np.array([1,1,-1,-1])
#get results from adaboost classifier
res = AdaBst_Alt(X, Y, tmax)
res.adaclassifier()
# print('hello')
Ypred = res.predict(X)
# print(Ypred)
# print("hello")
# print(Y)
# print(np.sum(Ypred == Y)/(Y.size))
plotres(X, Y, DR=res, axes=None, weights=None,stump = None)
#showing iterative results of stump vs ensemble decision real
# iterlist = list([0, 2, 4])
#
# fig, axes = plt.subplots(figsize=(8, len(iterlist) * 3),
#                          nrows=len(iterlist),
#                          ncols=2,
#                          sharex=True,
#                          dpi=100)
#
#
# _ = fig.suptitle('Decision boundaries by iteration')
# iter = 0
# for i in iterlist:
#
#     axstump, axboost = axes[iter]
#
#
#     # Plot weak learner
#     _ = axstump.set_title(f'Decision Stump t={i + 1}')
#     plotres(X=X, Y=Y,
#             stump=res.st[i], weights=res.smp_w[i],
#             axes=axstump)
#
#     # Plot strong learner
#     curres = dc(res)
#     curres.st = res.st[:i+1]
#     curres.st_w = res.st_w[:i+1]
#     _ = axboost.set_title(f'Aggregated Learner Using t={i + 1} Stumps')
#     plotres(X=X, Y=Y,
#             DR=curres, weights=curres.smp_w[i],
#             axes=axboost)
#
#     iter += 1

plt.tight_layout()
plt.show()

