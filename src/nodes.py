import utils as helper

class DecisionNode:
    '''Will record the Question to be asked, left and 
    right children (DecisionNode/PredictionNode)
    l_child contains true rows
    r_child contains false rows
    '''
    def __init__(self, question, nodeL, nodeR):
        self.question = question
        self.l_child = nodeL
        self.r_child = nodeR

class PredictionNode:
    '''This will contain label that is most common.
    Not to be intantiated directly.
    '''
    def __init__(self, rows):
        self.prediction = self._give_prediction(rows)
    
    def get_prediction(self):
        """Returns the lable or the average of the rows
        that was stored before
        @can make this method perticular to children, we
        want to fit model of linear regression in nodes
        later."""
        return self.prediction

class PredNodeClassify(PredictionNode):
    """Predction Node that classifies"""
    def __init__(self, rows):
        super().__init__(rows)    

    def _give_prediction(self, rows):
        print ("pritin the most probablie clss", helper.most_probable_label(rows))
        return helper.most_probable_label(rows)
        
class PredNodeRegress(PredictionNode):
    """Predction Node that regresses"""
    def __init__(self, rows):
        super().__init__(rows)    

    def _give_prediction(self, rows):
        vals = rows.iloc[:, -1]
        return vals.mean()
        