import matplotlib.pyplot as plt
import numpy as np 
from Tree import Tree


def main():
    print("This is a decsion tree test")
    val = input("Enter how many points in class 1 do you want:  ")
    val2 = input("Enter how many points in class 2 do you want: ")

    try:
        val_i = int(val)
        val_i2 = int(val2)
        if(val_i <= 0 or val_i2 <= 0):
            raise Exception()
    except Exception:
        print("Please input integers greater than zero only and run the program again")
        return ;

    plt.figure()
    npArray = np.ones((100,100))
    print("Click on the grid to make your dataset for class 1")

    plt.imshow(npArray)
    s_1 = plt.ginput(n = val_i,show_clicks = True)
    s_1 = np.array(s_1)

    print("Click on the grid to make your dataset for class 2")
    npArray = 0.5*np.ones((100,100))
    plt.imshow(npArray)

    s_2 = plt.ginput(n = val_i2,show_clicks = True)
    s_2 = np.array(s_2)
    data = np.concatenate((s_1, s_2), axis=0)

    labels_1 = np.ones((s_1.shape)[0])
    labels_2 = -1*np.ones((s_2.shape)[0])
    labels = np.concatenate((labels_1,labels_2), axis = 0)

    depth = input("Please input the depth of your decision tree: ")
    try:
        depth_i = int(depth)
        if(depth_i <= 0):
            raise Exception()
    except Exception:
        print("Please input integers greater than zero only and run the program again")
        return;

    # print(data)
    # print(labels)
    tree = Tree(None,depth_i,None,None,0)

    tree = Tree.make_tree(tree,data,labels,1,1)

    # tree.PrintTree()
    size = 400
    nx = np.linspace(0,100,size)
    ny = np.linspace(0,100,size)
    init_1 = 0
    init_2 = 0

    for i in nx:
        for j in ny:
            test_point = np.array([[i,j]])
            point_eval = Tree.evaluate_point(tree,test_point)
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
        


    plt.figure()
    plt.subplot(3,1,1)
    plt.scatter(data[:val_i ,0],data[:val_i ,1],labels + 2,linewidths=5,edgecolors='g')
    plt.scatter(data[val_i :,0],data[val_i :,1],labels + 2,linewidths=5,edgecolors='r')
    plt.title("Scatter plot of data")
    plt.xlabel("x1")
    plt.ylabel("x2")

    plt.subplot(3,1,3)
    plt.scatter(grid_1[:,0],grid_1[:,1],linewidths = 1,edgecolors='g',alpha=0.1)
    plt.scatter(grid_2[:,0],grid_2[:,1],linewidths = 1,edgecolors='r',alpha=0.1)
    plt.title("Scatter plot of decision boundaries")
    plt.xlabel("x1")
    plt.ylabel("x2")

    ccr = 0
    counter = 0
    for i in data:
        point = np.array([i])
        ccr += (labels[counter] == Tree.evaluate_point(tree,point))
        counter +=1
    
    print(counter)
    print("This is the CCR " , str(ccr/(counter)))

    plt.show()

if __name__ == "__main__":
    main()





