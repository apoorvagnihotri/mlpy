'''
Author: Apoorv Agnihotri

This implementation would be slow on larger datasets.
I would file the issues that make this implementation slow.

credits: https://github.com/random-forests/tutorials/blob/master/decision_tree.ipynb
'''

def unique_vals(rows, col):
    return set([row[col] for row in rows])

def label_counts(rows):
    """This only works for categoriacal data"""
    counts = {}
    for row in rows:
        label = row[-1]
        try:
            counts[label] += 1
        except KeyError:
            counts[label] = 0
    return counts

def most_probable_label(rows):
    '''Gives the label that comes the most number of times
    in the rows provided.
    @ find a counterpart for regression
    '''
    counts = label_counts(rows)
    best_label = -1
    max_count = 0
    for key in counts.keys():
        count = counts[key]
        if max_count <= count:
            max_count = count
            best_label = key
    return best_label

def is_numeric(value):
    return isinstance(value, int) or isinstance(value, float)

class Question:
    '''this class saves col_index and value
    for which a row would be considered
    greater than or not.
    '''
    def __init__(self, col_index, wedge):
        self.col_index = col_index
        self.wedge = wedge

    def is_satified(self, row):
        val = row[self.col_index]
        if is_numeric(val):
            return val >= self.wedge
        else:
            return val == self.wedge

def divide_on_question(rows, question):
    '''left list contains the true valued rows for the given
    question.
    @ again, a large duplication of data.
    '''
    left = []
    right = []
    for row in rows:
        if question.is_satified(row):
            left.append(row)
        else:
            right.append(row)
    return left, right

def gini_impurity(rows):
    '''Calculates the gini_impurity corresponding to
    the rows provided.
    '''
    counts = label_counts(rows)
    total = float(len(rows))
    gini_im = 1
    for count in counts.values():
        gini_im -= (count/total)**2
    return gini_im

def info_gain(left, right, current_impurity):
    '''Finds the information gain w.r.t. previous data in parent
    node.
    '''
    weight_left = float(len(left)) 
    weight_left /= len(left) + len(right)
    left_impurity = weight_left * gini_impurity(left)
    right_impurity = (1 - weight_left) * gini_impurity(right)
    new_impurity = current_impurity - left_impurity - right_impurity
    return new_impurity

def best_split(rows):
    '''Finds the best Question (using info_gain) that 
    splits the rows into left and right buckets.

    Returns a Question (records the col_index and the value.)
    @ bad performance when we have really large number of rows.
    In that case we can use a binary search for each coloumn to
    reduce the time complexity of the search for the best value
    to split.
    '''
    previous_impurity = gini_impurity(rows)
    BestInfoGain = 0 # find the best col and val to split the rows
    best_question = None
    for col in range(len(rows[0]) - 1): # last col is the label 
                                   # rows have same len
        for row in rows:
            val = row[col]         # this val may be the wedge
                                   # value for the question
            q = Question(col, val)
            left, right = divide_on_question(rows, q)
            if len(left) == 0 or len(right) == 0:
                continue           # ignore the case when no splitting
            InfoGain = info_gain(left, right, previous_impurity)
            # if best info gain crossed save state or save Question
            if InfoGain >= BestInfoGain:
                BestInfoGain = InfoGain
                best_question = q
    return BestInfoGain, best_question

class DecisionNode:
    '''Will record the Question to be asked, Parent (or not)
    and the left and right children (DecisionNode/PredictionNode)
    left_child contains true rows
    right_child contains false rows
    '''
    def __init__(self, question, nodeL, nodeR):
        self.question = question
        self.left_child = nodeL
        self.right_child = nodeR

class PredictionNode:
    '''This will contain label that is most common.
    @add support for regression.
    '''
    def __init__(self, rows, RorC):
        self.RorC = RorC
        if RorC == 'c':
            best_label = most_probable_label(rows)
            self.label = best_label
        else:
            pass

    def predict(self, row):
        if self.RorC == 'c':
            return self.label
        else:
            pass
            # return self.average_val of the rows 

    def average(self, rows):
        # summ 
        # self.average_val = summ
        pass

def classify(row, node):
    if isinstance(node, PredictionNode): # tree end
        return node.predict(row)
    if node.question.is_satified(row):
        return classify(row, node.left_child)
    else:
        return classify(row, node.right_child)

class DecisionTree:
    def __init__(self, max_depth, RorC):
        self.max_depth = max_depth
        self.RorC = RorC

    def build_tree(self, rows, depth):
        '''This function would be recursively called
        We first set the base case where we check if the
        the depth of tree is less than some number or
        we don't have any difference in labels
        @add support for regression
        '''
        if depth == self.max_depth: # forced leaf
            return PredictionNode(rows, self.RorC)

        gain, question = best_split(rows)
        if gain == 0:               # we have encountered a leaf
            return PredictionNode(rows, self.RorC)

        left, right = divide_on_question(rows, question)
        left_branch = self.build_tree(left, depth + 1)
        right_branch = self.build_tree(right, depth + 1)
        return DecisionNode(question, left_branch, right_branch)

    def train(self, rows):
        self.root = self.build_tree(rows, 0)

    def predict(self, row):
        return classify(row, self.root)

'''Learnings:
The way to think about object oriented programming is really
important, we see that the code above, inspired from the code
at https://github.com/random-forests/tutorials/blob/master/decision_tree.ipynb
makes us see that we can create objects for small small stuff
like Questions. This helps us in abstracting away the details from
and focus on the logic without worring about the implementation level
details (already taken care in the class of Question).

Thanks for making me grow by seeing your code @Josh @https://twitter.com/random_forests
'''