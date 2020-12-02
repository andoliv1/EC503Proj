import numpy as np
import matplotlib.pyplot as plt

## function below plots 2d adaboost results
def plotres(X, Y, weights = None , DR=None, axes = None):

    if axes is None:
        figure, axes = plt.subplots(figsize=(5, 5), dpi=100)

    n = X.shape[0]
    Xcls1 = []
    Xcls2 = []
    for i in range(n):
        if(Y[i] == 1):
            Xcls1 += [X[i, :]]
        elif(Y[i] == -1):
            Xcls2 += [X[i, :]]



    pad = 1
    x1_min, x1_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    x2_min, x2_max = X[:, 1].min() - pad, X[:, 1].max() + pad

    # if weights is not None:
    #     sizes = np.array(weights) * X.shape[0] * 100
    # else:
    #     sizes = np.ones(shape=X.shape[0]) * 100

    Xcls1 = X[Y == 1]
    Xcls2 = X[Y == -1]
    # axes.scatter(*Xcls1.T, s=sizes[Y == 1], marker='.', color='forestgreen')
    # axes.scatter(*Xcls2.T, s=sizes[Y == -1], marker='.', c='royalblue')
    axes.scatter(*Xcls1.T, marker='.', color='mediumvioletred')
    axes.scatter(*Xcls2.T, marker='.', c='royalblue')

    if DR:
        x1, x2 = np.meshgrid(np.arange(x1_min, x1_max, 0.01),
                             np.arange(x2_min, x2_max, 0.01))
        Ydec = DR.predict(np.c_[x1.ravel(), x2.ravel()])

        Ydec = Ydec.reshape(x1.shape)

        if sum(np.unique(Ydec)) == 0:
            class_regions = ['mediumvioletred', 'royalblue']
        elif sum(np.unique(Ydec)) == 1:
            class_regions = ['mediumvioletred']
        else:
            class_regions = ['royalblue']

        axes.contourf(x1, x2, Ydec, colors=class_regions, alpha=0.1)


    # axis.set_xlim(x1_min + 0.5, x1_max - 0.5)
    # axis.set_ylim(x2_min + 0.5, s2_max - 0.5)
    axes.set_xlabel('$x_1$')
    axes.set_ylabel('$x_2$')