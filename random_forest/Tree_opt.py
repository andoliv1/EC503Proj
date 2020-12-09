"""
Code for EC503 Project
References: (to be inputted)
"""
import random
import numpy as np
from numpy.random import randint
class Tree:

    def __init__(self,boundary,depth,left,right,val):
        """
        boundary: numpy array that holds in the first position the best split value and in the second position the best split dimension
        depth: integer that contains the height or depth of your tree
        left: Tree object that will continue to expand the tree
        right: Tree object that will continue to expand the tree
        val: integer which will hold the value of the node, this is only used if the node you are at is a leaf node
        """
        self.boundary = boundary
        self.depth = depth
        self.left = left
        self.right = right
        self.val = val

    @staticmethod
    def make_tree(tree,data,labels,want_random,num_random):
        """
        tree: Tree object that we wish to build
        data: numpy array that contains the data which your Tree will be built upon
        labels: numpy array that contains the labels of the data
        want_random: integer that specifies if you will parition each node based on a random subset of features
        num_random: what is the number of random features to partition each node. This is only valid if want_random == 1
        """
     
        
        # if your depth is 0 or there return and assign a value to your node which is the majority vote of the labels
        if(tree.depth == 0 or (True == (tree is None))):
            # print(data)
            # print(labels)
            tree.val = np.sign(np.sum(labels))
            # print(tree.val)
            return tree

        else:
            # if all of the data inside your node just belongs to one class than you don't need to keep splitting the data so just
            # assign the correct val to your node and return
            # print(labels)
            # print(type(labels))
            # print(data)
            if((labels == 1).all() or (labels == -1).all()):
                tree.val = labels[0]
                return tree
            
            # number of dimensions of a single data point 
            dimensions = data[0].size
            if(want_random == 1):
                # generate the subset of num_random features, Note: currently this is generating with replacement so I have to fix that
                dim_array =  random.sample(range(0,dimensions),num_random)
            
            else:
                # if you don't want random than you want to greedily split across all dimensions
                dim_array = np.linspace(0,dimensions-1,dimensions)

            # print(dim_array)
            # compute the optimal split and fetch the partitioned data after the split
            [best_split,best_impurity_score,best_dimension,data_left,data_right,labels_left,labels_right] = Tree.make_optimal_split(data,labels,dim_array)

            
            #record the best split onto your tree object
            tree.boundary = np.array([best_split,best_dimension])
            # if(data_left is not None):
            if(data_left is not None and data.size == data_left.size):
                #make the left tree and right tree recursively based on the same idea until you reach the depth wanted
                tree_left = Tree(None,tree.depth - 1,None,None,0)
                tree_left = Tree.make_tree(tree_left,data_left,labels_left,want_random,num_random)
                tree.left = tree_left
                tree.right = None
                return tree

            # if(data_right is not None):
            elif(data_right is not None and data.size == data_right.size):
                tree_right = Tree(None,tree.depth - 1,None,None,0)
                tree_right = Tree.make_tree(tree_right,data_right,labels_right,want_random,num_random)
                tree.right = tree_right
                tree.left = None
                return tree

            else:
                #make the left tree and right tree recursively based on the same idea until you reach the depth wanted
                tree_left = Tree(None,tree.depth - 1,None,None,0)
                tree_left = Tree.make_tree(tree_left,data_left,labels_left,want_random,num_random)
                tree.left = tree_left
            
                tree_right = Tree(None,tree.depth - 1,None,None,0)
                tree_right = Tree.make_tree(tree_right,data_right,labels_right,want_random,num_random)
                tree.right = tree_right
                return tree

        
  
    @staticmethod
    def make_optimal_split(data,labels,random_dimensions):
        """
        data: numpy array that contains the data which we want to split
        labels: numpy array that contains the labels of the data
        random_dimensions: dimensions which we will greedily split the data
        """

        # impurity_score_parent = Tree.compute_index(data,labels,0,0)

        # declare local variable that will keep track of the split
        best_impurity_score = 1
        best_split = 0
        best_dimension = 0
        data_left = None
        data_right = None
        labels_left = None
        labels_right = None

        num_positive = np.sum(labels == 1)
        num_negative = np.sum(labels == -1)

        # since we don't necessarily want to split the data at a point split it a little bit after a point, although this doesn't really matter

        #since we have discrete data we can find the best split by greedily splitting across each data point
        # split through all possible dimensions specified at each data point
        for t in random_dimensions:
            # print(t)
            data_t = data[:,t]
            indexing = np.argsort(data_t)
            new_data = data[indexing]
            new_labels = labels[indexing]
           
            b_1 = np.zeros(2)
            b_2 = np.zeros(2)
            b_2[0] = num_positive
            b_2[1] = num_negative
            
            # print(np.sum(b_2))
            for i in range(len(new_labels)):    
                # calculate the impurity score of the split
                if(new_labels[i] != 1 and new_labels[i] != - 1):
                    print("hooray")
                
                if(new_labels[i] == 1):
                    b_1[0] += 1
                    b_2[0] -= 1

                else:
                    # print(new_labels[i])
                    b_1[1] += 1
                    b_2[1] -= 1

                impurity_temp = Tree.compute_index(b_1,b_2)

                # if this is the best found split record it 
                if(best_impurity_score > impurity_temp):
                    best_impurity_score = impurity_temp
                    best_split = new_data[i,int(t)] 
                    best_dimension = int(t)
                    
                    data_left = new_data[:i+1]
                    data_right = new_data[i+1:]
                    labels_left = new_labels[:i+1]
                    labels_right = new_labels[i+1:]




        return [best_split,best_impurity_score,best_dimension,data_left,data_right,labels_left,labels_right]


    @staticmethod
    def compute_index(b_1,b_2):
        """
        data: numpy array that contains the data which we want to compute the impurity score on
        labels: numpy array that contains the labels of the data
        separator: integer that holds the value that data points will be compared to when computing the impurity score
        dimension: integer that holds the coordinate that data points will be indexed to compare their value at that coordinate with the 
        separator integer value
        """
        counter_1 = np.sum(b_1)
        counter_2 = np.sum(b_2)
        if(counter_1 < 0 or counter_2 < 0):
            print("yayeet")

        # if there are no points in b_1
        if(np.sum(b_1) == 0):
            impurity_score_1 = 0
        
        # if there are points than compute the gini index of b_1
        else:
            b_1 = b_1/(np.sum(b_1))
            impurity_score_1 = 1 - (b_1[0]**2) - (b_1[1]**2)
        
        # if there are no points in b_2
        if(np.sum(b_2) == 0):
            impurity_score_2 = 0

        # if there are points than compute the gini index of b_2
        else:
            b_2 = b_2/(np.sum(b_2))
            impurity_score_2 = 1 - (b_2[0]**2) - (b_2[1]**2)

        # compute the overall gini index of the two boxes added together
        impurity_score = counter_1*impurity_score_1 + counter_2*impurity_score_2
        impurity_score = impurity_score/(counter_1 + counter_2)

        return impurity_score

    

    @staticmethod
    def evaluate_point(tree,test_point):
        """
        tree: Tree object that you will parse through to find the evaluation of test_point
        test_point: numpy array containing the coordinates you want to have labeled
        """

        # if you have reached a node where you can't go deeper into the left or right node than it means you are at a leaf node
        # and shold return the value of your leaf node 
        if((tree.left is None) and (tree.right is None)):
            return tree.val
        
        elif(tree.left is None):
            x= Tree.evaluate_point(tree.right,test_point)
            return x
        
        elif(tree.right is None):
            x= Tree.evaluate_point(tree.left,test_point)
            return x

        # else continue parsing the tree until you are at a leaf node
        else:
            if(test_point[0,int(tree.boundary[1])] > tree.boundary[0]):
                tree_right = tree.right
                x= Tree.evaluate_point(tree.right,test_point)

                return x
            else:
                x= Tree.evaluate_point(tree.left,test_point)

                return x
    
    @staticmethod
    def evaluate_data(tree,data):
        """
        This function is just a wrapper that calls evaluate_point to evaluate whole arrays of data_points instead of a single 
        point 
        """
        init = 0
        for point in data:
            point = np.array([point])
            if(init == 0):
                evaluation = np.array([Tree.evaluate_point(tree,point)])
                init += 1
            else:
                evaluation = np.concatenate((evaluation,[Tree.evaluate_point(tree,point)]), axis=0)
        return evaluation

