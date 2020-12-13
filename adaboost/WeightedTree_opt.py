"""
Code for EC503 Project
Andreas Francisco, Yousuf Baker, Grayson Wiggins
"""

import numpy as np
from numpy.random import randint


class WeightedTree:

    def __init__(self, boundary, depth, left, right, val):

        self.boundary = boundary
        self.depth = depth
        self.left = left
        self.right = right
        self.val = val

    @staticmethod
    def make_tree(tree, data, labels, want_random, num_random, weight):
        
        if (tree.depth == 0 or (True == (tree is None))):
            c_0 = 0
            c_1 = 0
            counter = 0
            for i in labels:
                if (i == 1):
                    c_0 += weight[counter]

                else:
                    c_1 += weight[counter]
                counter += 1
            value = np.sign(c_0 - c_1)
            if (value == 0):
                i = randint(1, 2, 1)
                if (i == 1):
                    value = 1
                else:
                    value = -1

            tree.val = value

            return tree

        else:
            
            if ((labels == 1).all() or (labels == -1).all()):
                tree.val = labels[0]
                return tree

            
            dimensions = data[0].size
            if (want_random == 1):
                
                dim_array = randint(0, dimensions, num_random)

            else:
                
                dim_array = np.linspace(0, dimensions - 1, dimensions)


            [best_split, best_impurity_score, best_dimension,data_left,data_right,labels_left,labels_right,weight_left,weight_right] = WeightedTree.make_optimal_split(data, labels, dim_array,weight)

           
            tree.boundary = np.array([best_split, best_dimension])

            if (data_left is not None and data.size == data_left.size):
               
                tree_left = WeightedTree(None, tree.depth - 1, None, None, 0)
                tree_left = WeightedTree.make_tree(tree_left, data_left, labels_left, want_random, num_random,
                                                   weight_left)
                tree.left = tree_left
                tree.right = None
                return tree

            elif (data_right is not None and data.size == data_right.size):
                tree_right = WeightedTree(None, tree.depth - 1, None, None, 0)
                tree_right = WeightedTree.make_tree(tree_right, data_right, labels_right, want_random, num_random,
                                                    weight_right)
                tree.right = tree_right
                tree.left = None
                return tree

            else:
                
                tree_left = WeightedTree(None, tree.depth - 1, None, None, 0)
                tree_left = WeightedTree.make_tree(tree_left, data_left, labels_left, want_random, num_random,
                                                   weight_left)
                tree.left = tree_left

                tree_right = WeightedTree(None, tree.depth - 1, None, None, 0)
                tree_right = WeightedTree.make_tree(tree_right, data_right, labels_right, want_random, num_random,
                                                    weight_right)
                tree.right = tree_right
                return tree

        

    @staticmethod
    def make_optimal_split(data, labels, random_dimensions, weight):
        best_impurity_score = 1
        best_split = 0
        best_dimension = 0

        positive = (labels == 1)
        prob_positive = np.sum(weight[positive])
        negative = (labels == -1)
        prob_negative = np.sum(weight[negative])

        data_left = None 
        data_right = None
        labels_left = None 
        labels_right = None
        weight_left = None
        weight_right = None

        for t in random_dimensions:
            data_t = data[:,int(t)]
            indexing = np.argsort(data_t)
            new_data = data[indexing]
            new_labels = labels[indexing]
            new_weight = weight[indexing]
           
            b_1 = np.zeros(2)
            b_2 = np.zeros(2)
            b_2[0] = prob_positive
            b_2[1] = prob_negative

            for i in range(len(new_labels)):    
                
                if(new_labels[i] == 1):
                    b_1[0] += new_weight[i]
                    b_2[0] -= new_weight[i]
                else:
                    b_1[1] += new_weight[i]
                    b_2[1] -= new_weight[i]

                
                impurity_temp = WeightedTree.compute_index(b_1,b_2)

                
                if(best_impurity_score > impurity_temp):
                    best_impurity_score = impurity_temp
                    best_split = new_data[i,int(t)] 
                    best_dimension = int(t)
                    
                    data_left = new_data[:i+1]
                    data_right = new_data[i+1:]
                    labels_left = new_labels[:i+1]
                    labels_right = new_labels[i+1:]

                    weight_left = new_weight[:i+1]
                    weight_right = new_weight[i+1:]


        return [best_split,best_impurity_score,best_dimension,data_left,data_right,labels_left,labels_right,weight_left,weight_right]


    @staticmethod
    def compute_index(b_1,b_2):
        counter_1 = np.sum(b_1)
        counter_2 = np.sum(b_2)
        
        if(round(np.sum(b_1),10) == 0):
            impurity_score_1 = 0
        
       
        else:
            b_1 = b_1/(round(np.sum(b_1),10))
            impurity_score_1 = 1 - (b_1[0]**2) - (b_1[1]**2)
        
        
        if(round(np.sum(b_2),10) == 0):
            impurity_score_2 = 0

        
        else:
            b_2 = b_2/(round(np.sum(b_2),10))
            impurity_score_2 = 1 - (b_2[0]**2) - (b_2[1]**2)
           

        
        impurity_score = counter_1*impurity_score_1 + counter_2*impurity_score_2

        return impurity_score

    @staticmethod
    def evaluate_point(tree, test_point):

        if ((tree.left is None) and (tree.right is None)):
            return tree.val

        elif (tree.left is None):
            x = WeightedTree.evaluate_point(tree.right, test_point)
            
            return x

        elif (tree.right is None):
            x = WeightedTree.evaluate_point(tree.left, test_point)
    
            return x

        
        else:
            if (test_point[0, int(tree.boundary[1])] > tree.boundary[0]):
                tree_right = tree.right
                x = WeightedTree.evaluate_point(tree.right, test_point)
            
                return x
            else:
                x = WeightedTree.evaluate_point(tree.left, test_point)
                # print(x)
                return x

    @staticmethod
    def evaluate_data(tree, data):
        init = 0
        counter = 0

        for point in data:
            point2 = np.array([point])

            if (init == 0):
                evaluation = np.array([WeightedTree.evaluate_point(tree, point2)])
                init += 1
                counter += 1
            else:
                evaluation = np.concatenate((evaluation, [WeightedTree.evaluate_point(tree, point2)]), axis=0)
                counter += 1

        return evaluation