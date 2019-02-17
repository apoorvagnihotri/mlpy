from utils import prepend_ones_col

class Regression:
	def __init__(self, method='grad', regu=None):
		pass
	
	def train(self, X, y,
		      intercept=False
		      **kwarg):
	if intercept == False:
		self.X = prepend_ones_col(X)
		self.y = y

	self.find_theta()

