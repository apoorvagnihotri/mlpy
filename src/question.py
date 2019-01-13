import utils as helper

class Question:
    '''this class saves col_index and value
    for which rows would be considered
    greater than or not.
    '''
    def __init__(self, col_index, wedge):
        self.col_index = col_index
        self.wedge = wedge

    def divide_on_question(self, rows):
        '''left list contains the true valued rows for the given
        question.
        Assuming rows is a pandas Data Frame
        '''
        if helper.is_numeric(rows, self.col_index):
#             print (self.col_index, "is numeric")
            mask = rows.iloc[:, self.col_index] >= self.wedge
        else:
#             print (self.col_index, "isn't numeric")
            mask = rows.iloc[:, self.col_index] == self.wedge
        left = rows[mask]
        right = rows[~mask]
        return left, right
