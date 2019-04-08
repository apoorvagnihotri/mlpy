class SVM:
  """Implements a SVM Classifier"""
  def __init__(self,
               method='hard',
               kernel='linear'):
    self.method = method
    self.kernal = kernel

  def train(self, X, Y):
    '''Assumes that each row is a sample'''
    # Create two optimization variables.
    w = cp.Variable(X.shape[1])
    b = cp.Variable()

    # Create constraints.
    constraints = [*self._create_cons(Y, X, w, b)]
    print (constraints)

    # Form objective.
    obj = cp.Minimize((1/2)*cp.pnorm(w, p=2))

    # Form and solve problem.
    prob = cp.Problem(obj, constraints)
    prob.solve()  # Returns the optimal value.
    print("status:", prob.status)
    print("optimal value", prob.value)
    print("optimal var", w.value, b.value)
    self.prob = prob
    self.w = w
    self.b = b

  def _create_cons(self, Y, X, w, b):
    m = X.shape[0]
    exprs = []
    for i in range(m):
      expr = (Y[i] * (w @ np.squeeze(X[i, :]) + b))>= 1
      exprs.append(expr)
    return exprs

  def predict(self, X):
    y_pred = []
    for i in range(X.shape[0]):
      temp = self.w @ np.squeeze(X[i, :]) + self.b
      y_pred.append(float(temp.value))
    y_pred = np.asarray(y_pred)
    temp = y_pred >= 0
    temp2 = temp*1
    temp2 += np.logical_not(temp) * -1
    return temp2
