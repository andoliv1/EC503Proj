"""
Code for EC503 Project
References: (to be inputted)
"""

import numpy as np
from numpy.random import randint


class WeightedTree:

    def __init__(self, boundary, depth, left, right, val):
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
    def make_tree(tree, data, labels, want_random, num_random, weight):
        """
        tree: Tree object that we wish to build
        data: numpy array that contains the data which your Tree will be built upon
        labels: numpy array that contains the labels of the data
        want_random: integer that specifies if you will parition each node based on a random subset of features
        num_random: what is the number of random features to partition each node. This is only valid if want_random == 1
        """

        # if your depth is 0 or there return and assign a value to your node which is the majority vote of the labels
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
            # if all of the data inside your node just belongs to one class than you don't need to keep splitting the data so just
            # assign the correct val to your node and return
            if ((labels == 1).all() or (labels == -1).all()):
                tree.val = labels[0]
                return tree

            # number of dimensions of a single data point
            dimensions = data[0].size
            if (want_random == 1):
                # generate the subset of num_random features, Note: currently this is generating with replacement so I have to fix that
                dim_array = randint(0, dimensions, num_random)

            else:
                # if you don't want random than you want to greedily split across all dimensions
                dim_array = np.linspace(0, dimensions - 1, dimensions)

            # print(dim_array)
            # compute the optimal split and fetch the partitioned data after the split
            [best_split, best_impurity_score, best_dimension,data_left,data_right,labels_left,labels_right,weight_left,weight_right] = WeightedTree.make_optimal_split(data, labels, dim_array,weight)

            # record the best split onto your tree object
            tree.boundary = np.array([best_split, best_dimension])

            if (data_left is not None and data.size == data_left.size):
                # make the left tree and right tree recursively based on the same idea until you reach the depth wanted
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
                # make the left tree and right tree recursively based on the same idea until you reach the depth wanted
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
        """
        data: numpy array that contains the data which we want to split
        labels: numpy array that contains the labels of the data
        random_dimensions: dimensions which we will greedily split the data
        """

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

        # since we don't necessarily want to split the data at a point split it a little bit after a point, although this doesn't really matter

        #since we have discrete data we can find the best split by greedily splitting across each data point
        # split through all possible dimensions specified at each data point
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
                # calculate the impurity score of the split
                if(new_labels[i] == 1):
                    b_1[0] += new_weight[i]
                    b_2[0] -= new_weight[i]
                else:
                    b_1[1] += new_weight[i]
                    b_2[1] -= new_weight[i]

                # print(b_1)
                # print(b_2)
                impurity_temp = WeightedTree.compute_index(b_1,b_2)

                # if this is the best found split record it 
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

        # print(best_split)
        # print(best_dimension)

        return [best_split,best_impurity_score,best_dimension,data_left,data_right,labels_left,labels_right,weight_left,weight_right]


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
        # print(b_2)
        # if there are no points in b_1
        if(round(np.sum(b_1),10) == 0):
            impurity_score_1 = 0
        
        # if there are points than compute the gini index of b_1
        else:
            b_1 = b_1/(round(np.sum(b_1),10))
            impurity_score_1 = 1 - (b_1[0]**2) - (b_1[1]**2)
        
        # if there are no points in b_2
        if(round(np.sum(b_2),10) == 0):
            impurity_score_2 = 0

        # if there are points than compute the gini index of b_2
        else:
            b_2 = b_2/(round(np.sum(b_2),10))
            impurity_score_2 = 1 - (b_2[0]**2) - (b_2[1]**2)
           

        # compute the overall gini index of the two boxes added together
        impurity_score = counter_1*impurity_score_1 + counter_2*impurity_score_2

        return impurity_score

    @staticmethod
    def evaluate_point(tree, test_point):
        """
        tree: Tree object that you will parse through to find the evaluation of test_point
        test_point: numpy array containing the coordinates you want to have labeled
        """

        # if you have reached a node where you can't go deeper into the left or right node than it means you are at a leaf node
        # and shold return the value of your leaf node
        if ((tree.left is None) and (tree.right is None)):
            return tree.val

        elif (tree.left is None):
            x = WeightedTree.evaluate_point(tree.right, test_point)
            # print(x)
            return x

        elif (tree.right is None):
            x = WeightedTree.evaluate_point(tree.left, test_point)
            # print(x)
            return x

        # else continue parsing the tree until you are at a leaf node
        else:
            if (test_point[0, int(tree.boundary[1])] > tree.boundary[0]):
                tree_right = tree.right
                x = WeightedTree.evaluate_point(tree.right, test_point)
                # print(x)
                return x
            else:
                x = WeightedTree.evaluate_point(tree.left, test_point)
                # print(x)
                return x

    @staticmethod
    def evaluate_data(tree, data):
        """
        This function is just a wrapper that calls evaluate_point to evaluate whole arrays of data_points instead of a single
        point
        """
        init = 0
        counter = 0
        # print(data.shape)
        for point in data:
            point2 = np.array([point])
            # print(point)
            if (init == 0):
                evaluation = np.array([WeightedTree.evaluate_point(tree, point2)])
                init += 1
                counter += 1
            else:
                evaluation = np.concatenate((evaluation, [WeightedTree.evaluate_point(tree, point2)]), axis=0)
                counter += 1
            # print(counter)
        # print('yayeet')
        return evaluation