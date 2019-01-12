import os
import urllib.request as testfile

# testfile = urllib.request
path = ("https://archive.ics.uci.edu/ml/"
		"machine-learning-databases/00477/"
		"Real%20estate%20valuation%20data"
		"%20set.xlsx")
where = "../data/real_estate/"

if not os.path.exists(where):
    os.makedirs(where)

print ("Downloading...")

testfile.urlretrieve(path, where + "dataset.xlsx")

print ("Downloaded at", where)