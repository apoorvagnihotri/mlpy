def is_numeric(val):
    """Returns true if val is numeric"""
    return isinstance(val, int) or isinstance(val, float)

def label_counts(rows):
    """usefull only for categorical data"""
    counts = {}
    for row in rows:
        label = row[-1]
        try:
            counts[label] += 1
        except KeyError:
            counts[label] = 1
    return counts