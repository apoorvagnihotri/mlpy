import utils

class Criterion:
    """You should not be calling this class diretly.
    Initialize a derived class below.
    """
    def __init__(self):
        pass
        
    def info_gain(self, left, right, current_impurity):
        '''Finds the information gain w.r.t. previous data in parent
        node. The metric is decided by the child, STD, MSE, Gini or
        something else.
        '''
        weight_left = float(left.shape[0])
        weight_left /= left.shape[0] + right.shape[0]
        left_impurity = weight_left * self.impurity(left)
        right_impurity = (1 - weight_left) * self.impurity(right)
        new_impurity = current_impurity - left_impurity - right_impurity
        return new_impurity

class Gini(Criterion):
    def __init__(self):
        pass
    
    def impurity(self, rows):
        '''Calculates the gini_impurity corresponding to
        the rows provided.
        '''
        counts = utils.label_counts(rows)
        total = rows.shape[0]
        gini_im = 1
        for count in counts.values:
            gini_im -= (count/total)**2
        return gini_im

class Entropy(Criterion):
    def __init__(self):
        pass
    
    def impurity(self, rows):
        '''Calculates the entropy corresponding to
        the rows provided.
        '''
        counts = utils.label_counts(rows)
        total = rows.shape[0]
        entropy = 0
        for count in counts.values:
            entropy -= (count/total)*math.log((count/total), 2)
        return entropy

class STD(Criterion):
    def __init__(self):
        pass
    
    def impurity(self, rows):
        '''Calculates the STD on the y values
        corresponding to the rows provided.
        '''
        regress = rows.iloc[:, -1]
        return regress.std()
