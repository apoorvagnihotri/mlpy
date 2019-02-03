from utils import prepend_ones_col
import numpy as np

def normalEquationRegression(X, y):
    '''Assuming rows correspond to 
    samples
    '''
    X_new = prepend_ones_col(X)
    temp = np.dot(X_new.T, X_new)
    temp = np.linalg.inv(temp)
    temp = np.dot(temp, X_new.T)
    theta = np.dot(temp, y)
    return theta

def gradientDescentRegression(X, y, alpha=0.001, it=1000):
    X_new = prepend_ones_col(X)
    n, m = X_new.shape
    th = np.array([0.]*m)
    for i in range(it):
        yhat = np.matmul(X_new, th[:, None])
        e = y[:, None]-yhat
        weighted_e = X_new * e
        sumz = weighted_e.sum(axis=0)
        th = th + (2*alpha*(sumz))
    return th