import math
import numpy as np
from sklearn.tree import DecisionTreeClassifier

class AdaBoost():
    def __init__(self,
                 max_depth = 1,
                 num_of_trees = 5,
                 clf_def = DecisionTreeClassifier):
        self.clf_def = clf_def
        self.num_of_trees = num_of_trees
        self.max_depth = max_depth
        self.weights = []
        self.accs = []
    
    def train(self, X, y):
        # initial weights
        total = len(y)
        weights = [1./total]*total
        self.weights.append(weights)
        
        clfs, alphas = [], [] 
        for i in range(self.num_of_trees): # over all trees
            clf = self.clf_def(max_depth = self.max_depth)
            clf.fit(X, y, sample_weight = weights)
            preds = clf.predict(X)
            bad_preds = (preds != y)
            good_preds = (preds == y)
            err = bad_preds.sum()/total

            # calculating the weight for the classifier
            alpha = (1./2)*math.log((1-err)/err)

            # update the weights of the points
            g_weights = good_preds*weights*math.exp(-alpha)
            b_weights = bad_preds*weights*math.exp(alpha)

            # new weights ▼
            weights = g_weights + b_weights
            weights = weights/weights.sum() # normalize
            
            # saving weights, accs for animation ▼
            self.weights.append(weights)
            self.accs.append(good_preds.sum()/total)
            
            # saving clfs and their alphas for animation ▼
            clfs.append(clf)
            alphas.append(alpha)
        return clfs, alphas