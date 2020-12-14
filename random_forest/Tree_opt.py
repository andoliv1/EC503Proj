"""
Code for EC503 Project
Andreas Francisco, Yousuf Baker, Grayson Wiggins
"""
import random
import numpy as np
from numpy.random import randint
class Tree:

    #initializing tree object with the correct attributes
    def __init__(self,boundary,depth,left,right,val):
      
        self.boundary = boundary
        self.depth = depth
        self.left = left
        self.right = right
        self.val = val

    @staticmethod
    def make_tree(tree,data,labels,want_random,num_random):
       # if the depth is 0 then you don't want to continue splitting so just assign a value to the node I technically don't need to have the 
       # condition for None but since we are in the final stages of the project and don't want to reevaluate the whole code I will leave it there
        if(tree.depth == 0 or (True == (tree is None))):

            tree.val = np.sign(np.sum(labels))

            return tree
        #continue splitting
        else:
            # if the dataset only contains one class
            if((labels == 1).all() or (labels == -1).all()):
                tree.val = labels[0]
                return tree
            
            # want to random partition then select the dimensions on which to partition
            dimensions = data[0].size
            if(want_random == 1):
                
                dim_array =  random.sample(range(0,dimensions),num_random)
            
            else:
                
                dim_array = np.linspace(0,dimensions-1,dimensions)

            
            # make the optimal split and get the required parameters
            [best_split,best_impurity_score,best_dimension,data_left,data_right,labels_left,labels_right] = Tree.make_optimal_split(data,labels,dim_array)

            # assign the best split as a boundary of your tree object
            tree.boundary = np.array([best_split,best_dimension])
            
            # proceed to continue splitting recursively
            if(data_left is not None and data.size == data_left.size):
                
                tree_left = Tree(None,tree.depth - 1,None,None,0)
                tree_left = Tree.make_tree(tree_left,data_left,labels_left,want_random,num_random)
                tree.left = tree_left
                tree.right = None
                return tree

            
            elif(data_right is not None and data.size == data_right.size):
                tree_right = Tree(None,tree.depth - 1,None,None,0)
                tree_right = Tree.make_tree(tree_right,data_right,labels_right,want_random,num_random)
                tree.right = tree_right
                tree.left = None
                return tree

            else:
                
                tree_left = Tree(None,tree.depth - 1,None,None,0)
                tree_left = Tree.make_tree(tree_left,data_left,labels_left,want_random,num_random)
                tree.left = tree_left
            
                tree_right = Tree(None,tree.depth - 1,None,None,0)
                tree_right = Tree.make_tree(tree_right,data_right,labels_right,want_random,num_random)
                tree.right = tree_right
                return tree

        
  
    @staticmethod
    def make_optimal_split(data,labels,random_dimensions):
       # initialize variables to keep track of splits
        best_impurity_score = 1
        best_split = 0
        best_dimension = 0
        data_left = None
        data_right = None
        labels_left = None
        labels_right = None

        num_positive = np.sum(labels == 1)
        num_negative = np.sum(labels == -1)

        for t in random_dimensions:
            # sort the data with respect to the t_th dimension so that you can do any split across that dimenions in O(nk) time where n is 
            # the number of samples and k is the number of classes (which is just 2 for our code)
            data_t = data[:,t]
            indexing = np.argsort(data_t)
            new_data = data[indexing]
            new_labels = labels[indexing]
           
            b_1 = np.zeros(2)
            b_2 = np.zeros(2)
            b_2[0] = num_positive
            b_2[1] = num_negative
            
            # proceed to evaluate every possible split
            for i in range(len(new_labels)):    
                
                if(new_labels[i] == 1):
                    b_1[0] += 1
                    b_2[0] -= 1

                else:
                    b_1[1] += 1
                    b_2[1] -= 1

                impurity_temp = Tree.compute_index(b_1,b_2)

                # store the split if it is better than the current one
                if(best_impurity_score > impurity_temp):
                    best_impurity_score = impurity_temp
                    best_split = new_data[i,int(t)] 
                    best_dimension = int(t)
                    
                    data_left = new_data[:i+1]
                    data_right = new_data[i+1:]
                    labels_left = new_labels[:i+1]
                    labels_right = new_labels[i+1:]




        return [best_split,best_impurity_score,best_dimension,data_left,data_right,labels_left,labels_right]

    # compute the gini index 
    @staticmethod
    def compute_index(b_1,b_2):
        counter_1 = np.sum(b_1)
        counter_2 = np.sum(b_2)

        if(np.sum(b_1) == 0):
            impurity_score_1 = 0
        
       # get the gini impurity coefficient for each partition
        else:
            b_1 = b_1/(np.sum(b_1))
            impurity_score_1 = 1 - (b_1[0]**2) - (b_1[1]**2)
        
        
        if(np.sum(b_2) == 0):
            impurity_score_2 = 0

        
        else:
            b_2 = b_2/(np.sum(b_2))
            impurity_score_2 = 1 - (b_2[0]**2) - (b_2[1]**2)

        #compute the weighted average of the coefficient of each partition
        impurity_score = counter_1*impurity_score_1 + counter_2*impurity_score_2
        impurity_score = impurity_score/(counter_1 + counter_2)

        return impurity_score

    

    @staticmethod
    def evaluate_point(tree,test_point):
        #method for traversing through tree and predicting the value of the point

        #if you have reached a leaf return its value
        if((tree.left is None) and (tree.right is None)):
            return tree.val
        
        # if the the left is None then proceed to the right
        elif(tree.left is None):
            x= Tree.evaluate_point(tree.right,test_point)
            return x
        
        # if the right tree is None then proceed to the left
        elif(tree.right is None):
            x= Tree.evaluate_point(tree.left,test_point)
            return x

        # if both trees have something compare your point with the decision boundary and traverse accordingly
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
        # this is just a wrapper for the above function
        init = 0
        for point in data:
            point = np.array([point])
            if(init == 0):
                evaluation = np.array([Tree.evaluate_point(tree,point)])
                init += 1
            else:
                evaluation = np.concatenate((evaluation,[Tree.evaluate_point(tree,point)]), axis=0)
        return evaluation

