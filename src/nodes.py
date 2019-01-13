
class DecisionNode:
    '''Will record the Question to be asked, Parent (or not)
    and the left and right children (DecisionNode/PredictionNode)
    left_child contains true rows
    right_child contains false rows
    '''
    def __init__(self, question, nodeL, nodeR):
        self.question = question
        self.l_child = nodeL
        self.r_child = nodeR

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