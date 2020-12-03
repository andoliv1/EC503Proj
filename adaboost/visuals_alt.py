import numpy as np
import matplotlib.pyplot as plt
from WeightedTree import WeightedTree

## function below plots 2d adaboost results
def plotres(X, Y, weights = None , DR=None, axes = None, stump = None):

    
    
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
    print(DR)
    if DR:
        # print("got here")
        x1, x2 = np.meshgrid(np.arange(x1_min, x1_max, 0.05),
                             np.arange(x2_min, x2_max, 0.05))

        # print(x1)
        # print(x2)
        
        # print("This is x1 shape")
        # print(x1.shape)
        # print(np.c_[x1.ravel(), x2.ravel()])
        Ydec = DR.predict(np.c_[x1.ravel(), x2.ravel()])
        init_1 = 0
        init_2 = 0
        counter = 0
        x1 = x1.ravel()
        x2 = x2.ravel()
        for i in Ydec:
            if(i == 1):
                if(init_1 == 0):
                    label_1 = np.array([i])
                    grid_1 = np.array([[x1[counter],x2[counter]]])
                    init_1 += 1
                else:
                    label_1 = np.concatenate((label_1,[i]),axis = 0)
                    grid_1 = np.concatenate((grid_1,[[x1[counter],x2[counter]]]),axis = 0)
            else:
                if(init_2 == 0):
                    label_2 = np.array([i])
                    init_2 += 1
                    grid_2 = np.array([[x1[counter],x2[counter]]])
                else:
                    label_2 = np.concatenate((label_2,[i]),axis = 0)
                    grid_2 = np.concatenate((grid_2,[[x1[counter],x2[counter]]]),axis = 0)
            counter += 1
        
        print(grid_1)
        print(grid_2)

        plt.scatter(grid_2[:,0],grid_2[:,1],linewidths = 1,edgecolors='g',alpha=0.1)
        plt.scatter(grid_1[:,0],grid_1[:,1],linewidths = 1,edgecolors='r',alpha=0.1)
        plt.title("Scatter plot of decision boundaries")
        plt.xlabel("x1")
        plt.ylabel("x2")


        # # print("This is theresult of predict")
        # Ydec = Ydec.reshape(x1.shape)

        # if sum(np.unique(Ydec)) == 0:
        #     class_regions = ['mediumvioletred', 'royalblue']
        # elif sum(np.unique(Ydec)) == 1:
        #     class_regions = ['mediumvioletred']
        # else:
        #     class_regions = ['royalblue']

        # axes.contourf(x1, x2, Ydec, colors=class_regions, alpha=0.1)
    if stump:

        # x1, x2 = np.meshgrid(np.arange(x1_min, x1_max, 1),
        #                      np.arange(x2_min, x2_max, 1))

        # # print(x1)
        # # print(x2)
        
        # # print("This is x1 shape")
        # # print(x1.shape)
        
        # Ydec = WeightedTree.evaluate_data(stump,np.c_[x1.ravel(), x2.ravel()])
        # # print("This is theresult of predict")
        # Ydec = Ydec.reshape(x1.shape)
        # Ydec = Ydec
        # print(x1)
        # print(Ydec)

        # if sum(np.unique(Ydec)) == 0:
        #     class_regions = ['mediumvioletred', 'royalblue']
        # elif sum(np.unique(Ydec)) == 1:
        #     class_regions = ['mediumvioletred']
        # else:
        #     class_regions = ['royalblue']

        # axes.contourf(x1,x2, Ydec, colors=class_regions, alpha=0.1)
        size = 20

        nx = np.linspace(x1_min,x1_max,size)
        ny = np.linspace(x2_min,x2_max,size)

        init_1 = 0
        init_2 = 0

        for i in nx:
            for j in ny:
                test_point = np.array([[i,j]])
                point_eval = WeightedTree.evaluate_point(stump,test_point)
                if(point_eval == 1):
                    if(init_1 == 0):
                        grid_1 = np.array([[i,j]])
                    else:
                        grid_1 = np.concatenate((grid_1,[[i,j]]), axis = 0)
                    init_1 += 1
                else:
                    if(init_2 == 0):
                        grid_2 = np.array([[i,j]])
                    else:
                        grid_2 = np.concatenate((grid_2,[[i,j]]), axis = 0)
                    init_2 += 1
        
        plt.subplot(1,1,1)
        plt.scatter(grid_1[:,0],grid_1[:,1],linewidths = 1,edgecolors='g',alpha=0.1)
        plt.scatter(grid_2[:,0],grid_2[:,1],linewidths = 1,edgecolors='r',alpha=0.1)
        plt.title("Scatter plot of decision boundaries")
        plt.xlabel("x1")
        plt.ylabel("x2")

    # axis.set_xlim(x1_min + 0.5, x1_max - 0.5)
    # axis.set_ylim(x2_min + 0.5, s2_max - 0.5)
    axes.set_xlabel('$x_1$')
    axes.set_ylabel('$x_2$')
    print('fuck')