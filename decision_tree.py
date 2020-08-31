# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import math


def run_one_vs_all(class_labels):
    labels= []
    for clas in classes:
        labels.append(np.where(class_labels==clas, 1, -1))
    return labels

def pre_proccess():
    for attribute in Train_Data:
        Train_Data[attribute] = Train_Data[attribute].apply(pd.to_numeric, errors='coerce')
        Train_Data[attribute].fillna(Train_Data[attribute].median())
        treshhold = math.ceil(np.mean(Train_Data[attribute]))-1
        Train_Data[attribute] = np.where(Train_Data[attribute] > treshhold, 1, 0)

def entropy_func(c, n):
    return -(c*1.0/n)*math.log(c*1.0/n, 2)

def entropy_cal(c1, c2):
    if c1== 0 or c2 == 0: 
        return 0
    return entropy_func(c1, c1+c2) + entropy_func(c2, c1+c2)

def entropy_of_one_division(division): 
    s = 0
    n = len(division)
    classes = set(division)
    for c in classes:   # for each class, get entropy
        n_c = sum(division==c)
        e = n_c*1.0/n * entropy_cal(sum(division==c), sum(division!=c)) # weighted avg
        s += e
    return s, n

def get_entropy(y_predict, y_real):
    if len(y_predict) != len(y_real):
        return None
    n = len(y_real)
    s_true, n_true = entropy_of_one_division(y_real[y_predict]) # left hand side entropy
    s_false, n_false = entropy_of_one_division(y_real[~y_predict]) # right hand side entropy
    s = n_true*1.0/n * s_true + n_false*1.0/n * s_false # overall entropy, again weighted average
    return s





class DecisionTreeClassifier(object):
    def __init__(self, max_depth):
        self.depth = 0
        self.max_depth = max_depth
        self.trees = None
    
    def fit(self, x, y, par_node={}, depth=0):
        if par_node is None: 
            return None
        elif len(y) == 0:
            return None
        elif self.all_same(y):
            return {'val':y[0]}
        elif depth >= self.max_depth:
            return par_node
        else: 
            col, cutoff, entropy = self.find_best_split_of_all(x, y)
            y_left = y[x.iloc[:, col] < cutoff]
            y_right = y[x.iloc[:, col] >= cutoff]
            par_node = {'col': x[col], 'index_col':col, 'cutoff':cutoff, 'val': np.round(np.mean(y))}
            par_node['left'] = self.fit(x[x.iloc[:, col] < cutoff], y_left, {}, depth+1)
            par_node['right'] = self.fit(x[x.iloc[:, col] >= cutoff], y_right, {}, depth+1)
            self.depth += 1 
            self.trees = par_node
            return par_node
        
    def find_best_split_of_all(self, x, y):
        col = None
        min_entropy = 1
        cutoff = None
        for i in x:
            entropy, cur_cutoff = self.find_best_split(x[i], y)
            if entropy == 0:
                return i, cur_cutoff, entropy
            elif entropy <= min_entropy:
                min_entropy = entropy
                col = i
                cutoff = cur_cutoff
        return col, cutoff, min_entropy
    
    def find_best_split(self, col, y):
        min_entropy = 10
        n = len(y)
        for value in set(col):
            y_predict = col < value
            my_entropy = get_entropy(y_predict, y)
            if my_entropy <= min_entropy:
                min_entropy = my_entropy
                cutoff = value
        return min_entropy, cutoff
    
    def all_same(self, items):
        return all(x == items[0] for x in items)
    
    def predict(self, x):
        cur_layer = self.trees
        results = np.array([0]*len(x))
        for i in range(len(x)): 
            temp = self._get_prediction(x.iloc[i])
            if temp is None:
                temp = -1
            results[i] = temp 
        return results

    def _get_prediction(self, row):
        cur_layer = self.trees  # get the tree we build in training
        while cur_layer.get('cutoff'):   # if not leaf node
            if row[cur_layer['index_col']] < cur_layer['cutoff']:   # get the direction 
                cur_layer = cur_layer['left']
            else:
                cur_layer = cur_layer['right']
        else:   # if leaf node, return value
            return cur_layer.get('val')



def generate_trees(labels, train_data):
    decision_trees = []
    for class_labels in labels:
        print("constructing decsion tree for classes")
        tree = DecisionTreeClassifier(max_depth=10)
        tree.fit(train_data, class_labels[0:400])
        decision_trees.append(tree)
    print("decsion trees generated!")
    return decision_trees

def test(test_data, decision_trees):
    results = []
    for classifier in decision_trees:
        results.append(classifier.predict(test_data))
    return np.asarray(results)

def get_tree_pred(test_data, decision_trees):
    print("testing decsion trees")
    return test(test_data, decision_trees)

def predict_class(pred):
    predicted_class = []
    for c in pred.T:
        predicted_class.append(classes[np.argmax(c)])
    return np.asarray(predicted_class)

def accuracy(pred, label):
    acc = 0
    size = len(label)
    for i,v in enumerate(label):
        if pred[i] == v:
            acc += 1
    return acc/size
