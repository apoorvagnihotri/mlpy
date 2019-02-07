from sklearn.utils import shuffle
from utils import make_folds

def NestedCross:
    def __init__(self,
                 clf,
                 folds=4,
                 method='fwd_selection'):
        self.clf = clf
        self.folds = folds
        self.method = method
    
    def train(self, X, y):
        self.data = X, y
        folder = Folders(X, y,
                         self.folds,
                         method=self.method)
        tupl = folder.find_next_folds(last_err=None)
        X_train, y_train, X_val, y_val = tupl
        _select_folds()
        

class Folders:
    def __init__(self, X, y,
                 folds,
                 method='fwd_selection'):
        self.data = np.concatenate([X, y], axis=1)
        self.folds = folds
        self.method = method
        self.tol = 1
        self.cols = X.shape[1]

    def find_next_folds(self, lst_err):
        impv = 1000
        it = 0
        
        while (impv > self.tol or self.cols > it):
            if lst_err == None:
                folds_list = self._make_folds_wrapper(it)
            self._make_folds()
            it += 1

    def _select_cols(self):
        imp = self._find_imp()


    def _make_folds_wrapper(self, it): # @todo
        for i in range(self.cols)
            y = self.data[:, -1]
            temp = self.data[:, :i-it]
            temp2 = self.data[:, i+1:self.cols]
            rows = np.concatenate([temp, temp2, y], axis=1)
            fold_list = self._make_folds(rows)
        
        
    def _make_folds(self, rows):
        '''Make a number of folds with diven np'''
        train = shuffle(rows)
        last = train.shape[0]
        folds_list = []
        for i in range(self.folds):
            start = i*int(last/self.folds)
            end = (i+1)*int(last/self.folds)
            if end >= last:
                end = -1
            folds_list.append(train[start:end, :])
        return folds_list

    
def make_folds(dTrain, folds):
    '''Make a number of folds with diven pd'''
    train = shuffle(dTrain)
    last = train.shape[0]
    folds = []
    for i in range(folds):
        start = i*int(last/folds)
        end = (i+1)*int(last/folds)
        if end >= last:
            end = -1
        folds.append(train.iloc[start:end, :])
    return folds


## @todo, focusing on completing the assignment first, would work on this later.