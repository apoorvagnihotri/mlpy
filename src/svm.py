import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

class SVM:
  """Implements a SVM Classifier
  @todo supoort different kernels
  @todo support soft svm
  """
  def __init__(self,
               method='hard',
               kernel='linear',
               C=1e3):
    self.soft = True if method == 'soft' else False
    self.kernal = kernel
    self.C = C
    self.prob = None
    self.disctrete_out = True

  def train(self, X, Y):
    '''Assumes that each row is a sample'''
    # Create two optimization variables.
    w = cp.Variable(X.shape[1])
    b = cp.Variable()
    
    # Create constraints.
    constraints = self._create_cons(Y, X, w, b)

    # Form objective.
    obj = (1/2)*cp.norm(w,2)
    if self.soft:
      obj = obj/(self.C)
      for i in range(X.shape[0]):
        temp = cp.maximum(0, 1 - Y[i] * (w @ np.squeeze(X[i, :]) + b))
        obj = obj + temp
    obj = cp.Minimize(obj)

    # Form and solve problem.
    prob = cp.Problem(obj, constraints)
    prob.solve()  # Returns the optimal value.
    
    # storing for future accesses
    self.obj = obj
    self.constraints = constraints
    self.prob = prob
    self.w = w
    self.b = b

  def _create_cons(self, Y, X, w, b):
    '''For the case of soft svm we don't use any
    constrains'''
    m = X.shape[0]
    exprs = []
    if not self.soft:
      for i in range(m):
        expr = (Y[i] * (w @ np.squeeze(X[i, :]) + b))>= 1
        exprs.append(expr)
    return exprs
  
  def see_dual(self):
    if isinstance(self.prob, type(None)):
      raise ValueError("SVM has not been trained yet!")
    alphas = []
    for constraint in self.constraints:
      alphas.append(constraint.dual_value)
    return np.array(alphas)

  def predict(self, X):
    y_pred = []
    for i in range(X.shape[0]):
      temp = self.w @ np.squeeze(X[i, :]) + self.b
      y_pred.append(float(temp.value))
    y_pred = np.asarray(y_pred)
    self.y_pred = y_pred.copy() # storing for future access
    if not self.soft or self.disctrete_out:
      temp = y_pred >= 0
      y_pred = temp*1
      y_pred += np.logical_not(temp) * -1
    return y_pred
