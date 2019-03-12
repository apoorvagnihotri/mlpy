import pandas as pd
import time
import numpy as np
from nodes import *
import utils as helper
from sklearn.utils import shuffle

numerics = ['int16', 'int32', 'int64', 'float16',
            'float32', 'float64', int, float, 
            np.int16, np.int32, np.int64, np.float16,
            np.float32, np.float64]

def train_timer(func, **kwargs):
    ''' pass in the function you want to time,
    with the required keyword arguments'''
    t = Timer()
    func(**kwargs)
    val = t.lap()
    return val

class Timer:
    def __init__(self):
        self.start = time.time()

    def reset(self):
        self.start = time.time()

    def __del__(self):
        return self.lap()

    def lap(self):
        self.end = time.time()
        return self.end - self.start


def mode(a, axis=0):
    scores = np.unique(np.ravel(a))       # get ALL unique values
    testshape = list(a.shape)
    testshape[axis] = 1
    oldmostfreq = np.zeros(testshape)
    oldcounts = np.zeros(testshape)

    for score in scores:
        template = (a == score)
        counts = np.expand_dims(np.sum(template, axis),axis)
        mostfrequent = np.where(counts > oldcounts, score, oldmostfreq)
        oldcounts = np.maximum(counts, oldcounts)
        oldmostfreq = mostfrequent

    return mostfrequent, oldcounts


def rmse(pred, true):
    return np.sqrt(np.mean((pred-true)**2))

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

