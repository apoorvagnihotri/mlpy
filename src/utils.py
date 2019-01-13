import pandas as pd

def is_numeric(rows, col):
    """Returns true if val is numeric"""
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
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

