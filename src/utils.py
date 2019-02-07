import pandas as pd
import numpy as np
from nodes import *
import utils as helper
from sklearn.utils import shuffle

numerics = ['int16', 'int32', 'int64', 'float16',
            'float32', 'float64', int, float, 
            np.int16, np.int32, np.int64, np.float16,
            np.float32, np.float64]
    
def make_folds(dTrain, folds):
    '''Make a number of folds with diven pd'''
    train = shuffle(dTrain)
    last = train.shape[0]
    fold_list = []
    for i in range(folds):
        start = i*int(last/folds)
        end = (i+1)*int(last/folds)
        if end >= last:
            end = -1
        fold_list.append(train.iloc[start:end, :])
    return fold_list

def prepend_ones_col(X):
    """Expecting X to be a 2D array,
    returns ones prepended to X
    """
    n, m = X.shape
    return np.concatenate([np.ones((n, 1)), X], axis=1)

def parser(node, level=1):
    f = str(type(node)) == "<class 'nodes.DecisionNode'>"
    if f:
        if helper.is_num(node.question.wedge):
            operator = [">=", "<"]
        else:
            operator = ["=", "!="]

        ret = "Question: Is " + str(node.question.col_index) + \
        "th Column " + operator[0] + " " + str(node.question.wedge)+"\n"
        ret += level*"  " + "T: "+ parser(node.l_child, level+1)
        ret += level*"  " + "F: "+ parser(node.r_child, level+1)
    else:
        ret = "Predict: " + str(node.get_prediction()) + "\n"
    return ret

def is_num(val):
    return type(val) in numerics

def is_numeric(rows, col):
    """Returns true if val is numeric"""
    return rows.dtypes.values[col] in numerics

def label_counts(rows):
    """usefull only for categorical data"""
    counts = rows.iloc[:, -1].value_counts()
    return counts

def most_probable_label(rows):
    '''Gives the label that comes the most number of times
    in the rows provided.
    @ find a counterpart for regression
    '''
    counts = label_counts(rows)
    return counts.keys()[0]

def unique_vals(rows, col):
    return set([row[col] for row in rows])

