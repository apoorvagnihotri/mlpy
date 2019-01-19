import pandas as pd
import numpy as np
from nodes import *
import utils as helper
from sklearn.utils import shuffle

numerics = ['int16', 'int32', 'int64', 'float16',
            'float32', 'float64', int, float, 
            np.int16, np.int32, np.int64, np.float16,
            np.float32, np.float64]

def nested_cross(dTrain, depths, num_valid):
    folds = make_folds(dTrain, num_valid)
    # holding out test
    passed_folds = folds[:-1]
    k = best_k(passed_folds, depths)
    acc = 0
    for i in range(num_valid):
        curr_fold_train = [x for j,x in enumerate(folds) if j!=i] 
        curr_fold_test = folds[i]
        curr_fold_train = pd.concat(curr_fold_train)
        mydt = DecisionTreeClassifier(method='gini', max_depth=k)
        mydt.train(curr_fold_train)
        my_preds = np.squeeze(mydt.predict(curr_fold_test.iloc[:, :-1]).values)
        true = curr_fold_test.iloc[:, -1].values
        my_acc = ((true == my_preds).sum())/curr_fold_test.shape[0]
        acc += my_acc/num_valid
    return {"depth": k, "acc": acc}   

def make_folds(dTrain, num_valid):
    '''Make a number of folds with diven pd'''
    train = shuffle(dTrain)
    last = train.shape[0]
    folds = []
    for i in range(num_valid):
        start = i*int(last/num_valid)
        end = (i+1)*int(last/num_valid)
        if end >= last:
            end = -1
        folds.append(train.iloc[start:end, :])
    return folds

def best_k(folds, depths):
    '''return depth that maximizes the avg accuracy'''
    num_valid = len(folds)
    acc = {}
    for i in range(num_valid):
        curr_fold_train = [x for j,x in enumerate(folds) if j!=i] 
        curr_fold_validation = folds[i]
        curr_fold_train = pd.concat(curr_fold_train)
        for k in depths:
            mydt = DecisionTreeClassifier(method='gini', max_depth=k)
            mydt.train(curr_fold_train)
            my_preds = np.squeeze(mydt.predict(curr_fold_validation.iloc[:, :-1]).values)
            true = curr_fold_validation.iloc[:, -1].values
            my_acc = ((true == my_preds).sum())/curr_fold_validation.shape[0]
            if k in acc.keys():
                acc[k] += my_acc
            else:
                acc[k] = my_acc
    acc = {k: acc[k]/num_valid for k in acc.keys()}
    x = acc
    print (acc) # for showcasing
    sorted_by_value = sorted(x.items(), key=lambda kv: kv[1])
    return sorted(x[0] for x in sorted_by_value if sorted_by_value[-1][1] == x[1])[0]


def parser(node):
    print(parse(node))
    return

def parse(node, level=1):
    f = str(type(node)) == "<class 'nodes.DecisionNode'>"
    if f:
        if helper.is_num(node.question.wedge):
            operator = [">=", "<"]
        else:
            operator = ["=", "!="]

        ret = "Question: Is " + str(node.question.col_index) + \
        "th Column " + operator[0] + " " + str(node.question.wedge)+"\n"
        ret += level*"  " + "T: "+parse(node.l_child, level+1)
        ret += level*"  " + "F: "+parse(node.r_child, level+1)
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

