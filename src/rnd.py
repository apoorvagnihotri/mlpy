class Random:
    def __init__(self,
                 seed=0):
        self.last = seed
        
    def sample(self):
        X_next = self._sampler()
        self.last = X_next
        return X_next
        
    def random_between(self,
                       start,
                       end):
        x = self.sample()
        return (x % (end-start+1)) + start
    
class Uniform (Random):
    def __init__(self,
                 seed=0):
        super().__init__(seed)
    
    def _sampler(self):
        a = 2147483629
        c = 2147483587
        m = pow(2, 32) - 1
        lis = []
        X_next = ((self.last*a) + c)%m
        return X_next