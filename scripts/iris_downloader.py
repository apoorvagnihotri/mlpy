import os
import urllib.request as testfile

# testfile = urllib.request
path = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/"
where = "../data/iris/"

if not os.path.exists(where):
    os.makedirs(where)

print ("Downloading...")

testfile.urlretrieve(path + "Index", where + "Index")
testfile.urlretrieve(path + "iris.data", where + "iris.csv")
testfile.urlretrieve(path + "iris.names", where + "iris.names")

print ("Downloaded at", where)
