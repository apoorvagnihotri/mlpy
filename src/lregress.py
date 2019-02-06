import numpy as np
from autograd import grad
import torch
import autograd.numpy as anp
from torch.autograd import Variable
from utils import prepend_ones_col

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

def rms_lost(th, X, y):
    y_pred = anp.dot(X, th[:, None])
    error_square = anp.square(y[:, None]-y_pred)
    return error_square.sum()
rms_grad = grad(rms_lost)
def gradientDescentAutogradRegression(X, y, alpha=0.001, it=1000):
    X_new = prepend_ones_col(X)
    X_new = anp.array(X_new)
    n, m = X_new.shape
    th = anp.array([0.]*m)
    for i in range(it):
        dw = rms_grad(th, X_new, y)
        th = th - alpha*(dw)
    return th

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

