'''
Author: Apoorv Agnihotri

This implementation would be slow on larger datasets.
I would file the issues that make this implementation slow.

Coding style has been inspired from the below link,
https://github.com/random-forests/tutorials/blob/master/decision_tree.ipynb
Credits ^^^
'''

import criterion
from question import Question
import nodes

class DecisionTree:
    """you don't want to instantiate this
    object directly, use a derived class.
    """
    def __init__(self, 
                 method, 
                 max_depth, 
                 min_points):
        self.method = method
        self.max_depth = max_depth
        self.min_points = min_points
        self.criterion = self._make_metric_class()
        self.DecisionNode = nodes.DecisionNode
        self.root = None
        
    def train(self, rows):
        """Use this function to train the DT
        """
        self.root = self._build_tree(rows, 0)

    def _make_metric_class(self):
        """This function helps in modifying the criterion
        object on the fly, given classification or regression
        """
        if self.method=='gini':
            return criterion.Gini()

        elif self.method=='entropy':
            return criterion.Entropy()
        
        elif self.method=='std':
            return criterion.STD()
    
    def _best_split(self, rows):
        '''Finds the best Question (using info_gain) that 
        splits the rows into left and right buckets.

        Returns a Question (records the col_index and the value.)
        @ bad performance when we have really large number of rows.
        Can improve by first sorting and then doing a binary search
        '''
        previous_impurity = self.criterion.impurity(rows)
        BestInfoGain = 0 # find the best col and val to split the rows
        best_question = None
        num_rows, num_cols = rows.shape
        for col in range(num_cols - 1): # last col is the label 
                                       # rows have same len
            for row in range(num_rows):
                val = rows.iloc[row, col]         # this val may be the wedge
                                                  # value for the question
                q = Question(col, val)
                left, right = q.divide_on_question(rows)
                if left.shape[0] == 0 or right.shape[0] == 0:
                    continue           # ignore the case when no splitting
                InfoGain = self.criterion.info_gain(left, right, previous_impurity)
                # if best info gain crossed save state or save Question
                if InfoGain >= BestInfoGain:
                    BestInfoGain = InfoGain
                    best_question = q
                    if previous_impurity == InfoGain:
                        return BestInfoGain, best_question # already best
        return BestInfoGain, best_question
    
    def _build_tree(self, rows, depth):
        '''This function would be recursively called
        We first set the base case where we check if the
        the depth of tree is less than some number or
        we don't have any difference in labels
        @add support for regression
        '''
        if depth == self.max_depth: # tree depth reached
            return self.PredictionNode(rows)
        
        # return prediction node if min_points crossed
        if not isinstance(self.min_points, type(None)):
            if min_points <= rows.shape[0]:
                return self.PredictionNode(rows)

        gain, question = self._best_split(rows)
        if gain == 0:               # we have encountered a leaf
            return self.PredictionNode(rows)

        left, right = question.divide_on_question(rows)
        left_branch = self._build_tree(left, depth + 1)
        right_branch = self._build_tree(right, depth + 1)
        return self.DecisionNode(question, left_branch, right_branch)
    
    def predict(self, rows):
        # for each row in df, apply self._predict_one
        return rows.apply(self._predict_one, axis=1).to_frame()

    def _predict_one(self, row, node=None):
        if isinstance(node, type(None)):
            node = self.root
        if isinstance(node, self.PredictionNode): # tree end
            return node.get_prediction()
        # Else we have a DecisionNode
        if node.question.is_satified(row):
            return self._predict_one(row, node.l_child)
        else:
            return self._predict_one(row, node.r_child)
    

class DecisionTreeClassifier(DecisionTree):
    """Instantiate this class for a Decision
    Tree for classification tasks.
    """
    def __init__(self,
                 method="gini",
                 max_depth=2,
                 max_points=None):
        self.PredictionNode = nodes.PredNodeClassify    # Prediction node is 
                                            # different in case of Regression
        super().__init__(method,
                         max_depth,
                         max_points)

class DecisionTreeRegression(DecisionTree):
    """Instantiate this class for a Decision
    Tree for Regression tasks.
    """
    def __init__(self,
                 method="std",
                 max_depth=2,
                 max_points=None):
        self.PredictionNode = nodes.PredNodeRegress    # Prediction node is 
                                                 # different in case of Regression
        super().__init__(method,
                         max_depth,
                         max_points)

