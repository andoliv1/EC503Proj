"""
Code for EC503 Project
Andreas Francisco, Yousuf Baker, Grayson Wiggins
"""
import random
import numpy as np
from numpy.random import randint
class Tree:

    def __init__(self,boundary,depth,left,right,val):
      
        self.boundary = boundary
        self.depth = depth
        self.left = left
        self.right = right
        self.val = val

    @staticmethod
    def make_tree(tree,data,labels,want_random,num_random):
       
        if(tree.depth == 0 or (True == (tree is None))):

            tree.val = np.sign(np.sum(labels))

            return tree

        else:
        
            if((labels == 1).all() or (labels == -1).all()):
                tree.val = labels[0]
                return tree
            
            
            dimensions = data[0].size
            if(want_random == 1):
                
                dim_array =  random.sample(range(0,dimensions),num_random)
            
            else:
                
                dim_array = np.linspace(0,dimensions-1,dimensions)

            
            [best_split,best_impurity_score,best_dimension,data_left,data_right,labels_left,labels_right] = Tree.make_optimal_split(data,labels,dim_array)

            
            
            tree.boundary = np.array([best_split,best_dimension])
            
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
            # print(t)
            data_t = data[:,t]
            indexing = np.argsort(data_t)
            new_data = data[indexing]
            new_labels = labels[indexing]
           
            b_1 = np.zeros(2)
            b_2 = np.zeros(2)
            b_2[0] = num_positive
            b_2[1] = num_negative
            
            
            for i in range(len(new_labels)):    
                
                if(new_labels[i] != 1 and new_labels[i] != - 1):
                    print("hooray")
                
                if(new_labels[i] == 1):
                    b_1[0] += 1
                    b_2[0] -= 1

                else:
                    b_1[1] += 1
                    b_2[1] -= 1

                impurity_temp = Tree.compute_index(b_1,b_2)

        
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
        counter_1 = np.sum(b_1)
        counter_2 = np.sum(b_2)

        if(np.sum(b_1) == 0):
            impurity_score_1 = 0
        
       
        else:
            b_1 = b_1/(np.sum(b_1))
            impurity_score_1 = 1 - (b_1[0]**2) - (b_1[1]**2)
        
        
        if(np.sum(b_2) == 0):
            impurity_score_2 = 0

        
        else:
            b_2 = b_2/(np.sum(b_2))
            impurity_score_2 = 1 - (b_2[0]**2) - (b_2[1]**2)

        impurity_score = counter_1*impurity_score_1 + counter_2*impurity_score_2
        impurity_score = impurity_score/(counter_1 + counter_2)

        return impurity_score

    

    @staticmethod
    def evaluate_point(tree,test_point):
    
        if((tree.left is None) and (tree.right is None)):
            return tree.val
        
        elif(tree.left is None):
            x= Tree.evaluate_point(tree.right,test_point)
            return x
        
        elif(tree.right is None):
            x= Tree.evaluate_point(tree.left,test_point)
            return x

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
        
        init = 0
        for point in data:
            point = np.array([point])
            if(init == 0):
                evaluation = np.array([Tree.evaluate_point(tree,point)])
                init += 1
            else:
                evaluation = np.concatenate((evaluation,[Tree.evaluate_point(tree,point)]), axis=0)
        return evaluation

