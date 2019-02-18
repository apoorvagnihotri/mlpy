import numpy as np
from autograd import grad
import torch
import autograd.numpy as anp
from torch.autograd import Variable
from utils import prepend_ones_col


# cost for autograd
def lasso_cost(th, X, y, lmbd):
    y_pred = anp.dot(X, th[:, None])
    error_square = anp.square(y[:, None]-y_pred)
    lasso_error = anp.linalg.norm(th, ord=1) * (lmbd**2)
    return error_square.sum() + lasso_error
lasso_grad = grad(lasso_cost)
def gradientDescentAutogradRegression(X, y, alpha=0.001, lmbd=0.1, it=300):
    X_new = prepend_ones_col(X)
    X_new = anp.array(X_new)
    y = anp.array(y)
    n, m = X_new.shape
    th = anp.random.rand(m)
    for i in range(it):
        dw = lasso_grad(th, X_new, y, lmbd)
        th = th - alpha*(dw)
    return th




def normalEquationRidgeRegression(X, y, lmbd=0.1):
    '''Assuming rows correspond to samples'''
    X_new = prepend_ones_col(X)
    _, dim = X_new.shape
    temp = np.dot(X_new.T, X_new) + np.eye(dim)*(lmbd**2)
    temp = np.linalg.inv(temp)
    temp = np.dot(temp, X_new.T)
    theta = np.dot(temp, y)
    return theta

def coodrdinateDescentRegression(X, y, it=100):
    X_new = prepend_ones_col(X)
    n, m = X_new.shape
    th = np.array([0.]*m)
    
    for iteration in range(it): # for a num of iterations
        for i in range(m): # for num of thetas
            # calculate y_hat without th_i*X_i
            y_hat_red = np.dot(np.delete(X_new, i, axis=1), np.delete(th, i, axis=0))
            temp = np.dot(X_new[:,i].T, (y - y_hat_red))
            th[i] = temp / np.matmul(X_new[:, i].T, X_new[:, i])
    return th

def coodrdinateDescentLasso(X, y, lmbd=0.2, it=100):
    X_new = prepend_ones_col(X)
    n, m = X_new.shape
    th = np.array([0.]*m)
    
    for iteration in range(it): # for a num of iterations
        for i in range(m): # for num of thetas
            # calculate y_hat without th_i*X_i
            y_hat_red = np.dot(np.delete(X_new, i, axis=1), np.delete(th, i, axis=0))
            p_i = np.dot(X_new[:,i].T, (y - y_hat_red))
            z_i = np.matmul(X_new[:, i].T, X_new[:, i])
            const = (lmbd**2)/2
            if p_i < -const:
                th[i] = (p_i + const)/z_i
            elif p_i >= -const and p_i <= const:
                th[i] = 0
            else: # p_i > const
                th[i] = (p_i - const)/z_i
    return th

def sgdRegression(X, y, alpha = 0.001, it=100):
    X_new = prepend_ones_col(X)
    n, m = X_new.shape
    th = np.zeros(m)
    for i in range(it):
        for j in range(n):
            yhat = (X_new[j, :] * th).sum()
            e = y[j]-yhat
            weighted_e = X_new[j, :] * e
            th = th + (2*alpha*(weighted_e))
    return th
    
def normalEquationRegression(X, y):
    '''Assuming rows correspond to samples'''
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

# cost for pytorch
def cost(th, X, y):
    yhat = torch.mm(X, th.double())
    return torch.sum((y - yhat)**2)
def gradientDescentPytorchRegression(X, y, alpha=0.001, it=1000):
    X_new = torch.from_numpy(prepend_ones_col(X))
    y = torch.from_numpy(y[:, None])
    n, m = X_new.shape
    th = Variable(torch.rand(m, 1), requires_grad=True)
    for i in range(it):
        error = cost(th, X_new, y)
        error.backward()
        th = Variable(th - alpha * th.grad, requires_grad=True)
    nump = th.detach().numpy()
    return nump.squeeze()

