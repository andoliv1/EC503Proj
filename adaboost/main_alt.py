from adaboost_alt import AdaBst_Alt
import numpy as np
import matplotlib.pyplot as plt
from visuals_alt import plotres
from copy import deepcopy as dc
from sklearn.datasets import make_gaussian_quantiles
from WeightedTree import WeightedTree


#initialize sample dataset
mu = 0
sigma = 1
n = 20

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
Y = Y*2-1
tmax = 50

#get results from adaboost classifier
res = AdaBst_Alt(X, Y, tmax)
res.adaclassifier()
print('hello')
plotres(X, Y, DR=res, axes=None, weights=None)
ts = np.arange(1, tmax+1)
ccr = np.zeros(shape=tmax)
for t in range(tmax):
    curres = dc(res)
    curres.st = res.st[:t]
    curres.st_w = res.st_w[:t]
    Ypred = curres.predict(X)
    ccr[t] = sum(Ypred == Y) / Y.size

plt.figure()
plt.plot(ts, ccr)
plt.title('Number of Weak Learners in Ensemble Learner (n) vs CCR')
plt.xlabel("n")
plt.ylabel("CCR")
#showing iterative results of stump vs ensemble decision real
iterlist = list([0, 25, 49])

fig, axes = plt.subplots(figsize=(8, len(iterlist) * 3),
                         nrows=len(iterlist),
                         ncols=2,
                         sharex=True,
                         dpi=100)


_ = fig.suptitle('Decision boundaries by iteration')
iter = 0
for i in iterlist:

    axstump, axboost = axes[iter]


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

