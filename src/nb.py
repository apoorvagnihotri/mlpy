'''
Author: Apoorv Agnihotri

Naive Bayes
'''
# Inspired from https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/
import operator
import random
import math

class Gaussian:
    def __init__(self):
        pass

    def calculateLogProbability(self, x, mean, stdev):
        '''x, mean and stdev are vectors of resp. fea, returns a scalar'''
        num  = -np.square(x-mean)
        power = num/(2*np.square(stdev))
        exponent = np.exp(power)
        return np.sum(np.log((1 / (np.sqrt(2*np.pi) * stdev)) * exponent))

class NaiveBayes:
    def __init__(self, method='gaus'):
        if method == 'gaus':
            self.prp = Gaussian()
        
    def train(self, X, y):
        self.dataset = np.concatenate([X, y[:, None]], axis=1)
        self.summeries = self.summarizeByClass(self.dataset)
    
    def separateByClass(self, dataset):
        separated = {}
        for i in range(dataset.shape[0]):
            vector = dataset[i, :]
            if (vector[-1] not in separated):
                separated[vector[-1]] = []
            separated[vector[-1]].append(vector)
        return separated

    def summarize(self, dataset):
        attr_mean = []
        attr_std = []
        for i in range(dataset.shape[1]-1):
            attr = dataset[:, i]
            attr_mean.append(np.mean(attr))
            attr_std.append(np.std(attr))
        return (np.array(attr_mean), np.array(attr_std))
    
    def summarizeByClass(self, dataset):
        separated = self.separateByClass(dataset)
        attrs_mean_std_per_class = {}
        for classValue, instances in separated.items():
            attrs_mean_std_per_class[classValue] = self.summarize(np.array(instances))
        return attrs_mean_std_per_class # is a dict with tuple of vectors

    def calculateClassProbabilities(self, x):
        '''Calculates the class probabilities for x given theta.'''
        summaries = self.summeries
        probabilities = {}
        for classValue, classSummaries in summaries.items():
            # expanding classSummaries -> (mean_vct, std_vct)
            probabilities[classValue] = self.prp.calculateLogProbability(x, *classSummaries)
        return probabilities

    def predict(self, xs, conf=False):
        preds = []
        probs = []
        for x in xs:
            probabilities = self.calculateClassProbabilities(x)
            bestLabel = max(probabilities.items(), key=operator.itemgetter(1))[0]
            preds.append(bestLabel)
            probs.append(probabilities)
        if conf:
            return np.array(preds), np.array(probs)
        else:
            return np.array(preds)
