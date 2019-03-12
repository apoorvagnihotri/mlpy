import numpy as np

class DMetric:
    """You should not be calling this class diretly.
    Initialize a derived class below.
    """
    def __init__(self):
        pass
        
    def calc_dist(self, targets, point):
        '''One Should not call unspecialized DMetric
        class's `calc_dist` function
        '''
        raise NotImplementedError("Subclasses should implement this!")


class Euclidian(DMetric):
    def __init__(self):
        pass
    
    def calc_dist(self, targets, point):
        '''Calculates the Euclidian distance between
        two points
        '''
        point = np.expand_dims(point, axis=0)
        offsets = targets - point
        dist = np.linalg.norm(offsets, ord=2, axis=1)
        return dist

class Manhattan(DMetric):
    def __init__(self):
        pass
    
    def calc_dist(self, targets, point):
        '''Calculates the Manhattan distance between
        two points
        '''
        point = np.expand_dims(point, axis=0)
        offsets = targets - point
        dist = np.linalg.norm(offsets, ord=1, axis=1)
        return dist

class Cosine(DMetric):
    def __init__(self):
        pass
    
    def calc_dist(self, targets, point):
        '''Calculates `1 - Cosine Similarity` between
        two points
        '''
        assert (point is not None)
        point = np.expand_dims(point, axis=0)
        num = np.sum(targets * point, axis=1)
        t_magnitude = np.linalg.norm(targets, ord=2, axis=1)
        p_magnitude = np.linalg.norm(point, ord=2)
        den = t_magnitude * p_magnitude
        return num/den
