# =============================================================================
# Types and constants
# =============================================================================

CRITERIA_CLF = {"gini": _criterion.Gini, "entropy": _criterion.Entropy}
CRITERIA_REG = {"mse"}

class DecisionTree:
    def __init__():
        self.trained_tree # the trained model
        self.D = # number of training examples.
        self.max_depth = # height of the decision tree
        self.RorC = # Regression or Classification
        self.root # root node
        self.criterion # IG, Gini
        self.Eta # variance (of y) at when we stop splitting
        self.D_limit # number of data points at which we
                    # stop spliting subnode.
        self.model # model to use internally inside a leaf
                   # (for regression)



    def train(X_train):


'''
brief: _split the dataset provided on the
       feature x, with value v
@param data
@param x the feature on which we 
         would like to split
@param v the value with which we want to split
'''
    def _split():

    def _get_prediction_node():

    def _stopping_criteria():
        # if the data ariving is really small or var(y) is smaller than some eta

    def _build_subtree():

    def _get_IG():

    def _get_split_variable_index():

    def predict(X_test):


class DecisionTreeClassifier(DecisionTree):
    pass
class DecisionTreeRegression(DecisionTree):
    def __init__(self,
                 criterion="mse",
                 # splitter="best",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features=None,
                 random_state=None,
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 presort=False):