import urllib.request as testfile

# testfile = urllib.request
path = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/"
where = "../data/iris/"

print ("Downloading...")

testfile.urlretrieve(path + "Index", where + "Index")
testfile.urlretrieve(path + "bezdekIris.data", where + "bezdekIris.data")
testfile.urlretrieve(path + "iris.data", where + "iris.data")
testfile.urlretrieve(path + "iris.names", where + "iris.names")

print ("Downloaded at", where)