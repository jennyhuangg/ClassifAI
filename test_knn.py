"""
Implements the k-nearest neighbors algorithm for bag-of-words.
"""

import math
import operator
from math import log
import time
import generatedata.feature_helper as feature_helper

# number of trials
TRIALS = 5

class KNNTextClassifier:  
    
    def __init__(self):
        self.k = 5 
        self.alpha = 0
        self.distForm = 0 # 0 Euclidean, 1 Manhattan, 2 Maximum

    # setter functions

    def setK(self, k):
        self.k = k

    def setAlpha(self, alpha):
        self.alpha = alpha

    def setDistForm(self, d):
        self.distForm = d

    # distance metric functions

    def euclideanDistance(self, point1, point2):
        distance = 0
        for i in range(len(point1)):
            distance += pow((point1[i] - point2[i]), 2)
        return math.sqrt(distance)

    def manhattanDistance(self, point1, point2):
        distance = 0
        for i in range(len(point1)):
            distance += abs(point1[i] - point2[i])
        return distance

    def maxDistance(self, point1, point2):
        distance = 0
        for i in range(len(point1)):
            distance = max(distance, abs(point1[i] - point2[i]))
        return distance

    # calculates the k closest users in the training set to a test instance
    def getNeighbors(self, testInstance):
        # get distance metric
        distances = []
        for i in range(len(self.counts)):
            if self.distForm == 0:
                dist = self.euclideanDistance(testInstance, self.counts[i])
            elif self.distForm == 1:
                dist = self.manhattanDistance(testInstance, self.counts[i])
            else:
                dist = self.maxDistance(testInstance, self.counts[i])
            distances.append((dist, i))
        distances.sort(key=operator.itemgetter(0))
        neighbors = []
        for x in range(self.k):
            neighbors.append(distances[x][1])
        return neighbors
    
    # predicts class of a test instance given their neighbors
    def getPrediction(self, neighbors):
        counts = [0, 0]
        for n in neighbors:
            label = self.labels[n]
            counts[label] += 1
        return counts[1] > counts[0]

    # loads in the traning set file and builds transing set data structures
    def buildTrainingSet(self, infile):
        #  2D list, where each list corresponds to a user and contains the frequency of the word for that user
        self.counts = []
        # the correct labels per user
        self.labels =[]
        # the dictionary, with the word as the key and the index as the value.
        self.words = {}
        # total count per word in all the training set
        self.totals = {}

        with open(infile) as f:
            content = f.readlines()
        content = [x.strip() for x in content]

        ind = 0
        for line in content:
            line = line.lower()
            words = line.split(" ")
            label = int(words.pop(0))
            self.labels.append(label)

            counts = [self.alpha] * len(self.words)
            for word in words:
                if word not in self.words:
                    self.words[word] = ind
                    ind += 1
                    for c in self.counts:
                        c.append(self.alpha)
                    counts.append(self.alpha)
                counts[self.words[word]] += 1. / len(words)
                try:
                    self.totals[self.words[word]] += 1
                except:
                    self.totals[self.words[word]] = 1
            self.counts.append(counts)

        # using the term frequency - inverse document frequency 
        for c in self.counts:
            for i in range(len(c)):
                c[i] *= log(len(content) / float(self.totals[i]))

    # builds the test set from file, makes inferences, and checks accuracy
    def testSet(self, infile):
        # build test set
        testcounts = []
        testlabels = []

        # true positive, false positive, false negative, true negative
        confusion_matrix = [0] * 4

        with open(infile) as f:
            testSet = f.readlines()
        testSet = [x.strip() for x in testSet]

        for line in testSet:
            line = line.lower()
            words = line.split(" ")
            label = int(words.pop(0))
            testlabels.append(label)
            counts = [self.alpha] * len(self.words)

            for word in words:
                if word in self.words:
                    counts[self.words[word]] += 1. / len(words)
            testcounts.append(counts)

        for count in testcounts:
            for i in range(len(count)):
                count[i] *= log(len(testSet) / float(self.totals[i]))
        
        # get accuracy
        correct = 0
        for x in range(len(testSet)):
            neighbors = self.getNeighbors(testcounts[x])
            prediction = self.getPrediction(neighbors)
            if testlabels[x]:
                if prediction:
                    correct += 1
                    confusion_matrix[0] += 1
                else:
                    confusion_matrix[2] += 1
            else:
                if prediction:
                    confusion_matrix[1] += 1
                else:
                    correct += 1
                    confusion_matrix[3] += 1
        
        print "confusion matrix:", confusion_matrix
        specificity = float(confusion_matrix[3]) / (confusion_matrix[3] + confusion_matrix[1])
        sensitivity = float(confusion_matrix[0]) / (confusion_matrix[0] + confusion_matrix[2])
        precision = float(confusion_matrix[0]) / (confusion_matrix[0] + confusion_matrix[1])
        accuracy = (correct / float(len(testSet))) * 100.0
        stats = [accuracy, specificity, sensitivity, precision]
        print "stats: ", stats
        return stats

def runMultipleTests(c):
    '''
    Performs testing on the different files
    '''
    stats_total = [0.] * 4
    runtime = 0.
    for i in range(TRIALS):
        start_time = time.time()
        print "Building training set..."
        c.buildTrainingSet('generatedata/text/training_tweets' + str(i))
        #print len(c.words), "words in dictionary"
        temp_acc = c.testSet('generatedata/text/test_tweets' + str(i))
        print("--- %s seconds ---" % (time.time() - start_time))
        runtime +=(time.time() - start_time)
        #print "Accuracy on validation set:", temp_acc[0]
        stats_total = [x + y for x, y in zip(stats_total, temp_acc)]

    avgStats = [x / TRIALS for x in stats_total]
    print "Overall Average Stats: ", avgStats
    print "Average run time: ", runtime / TRIALS

if __name__ == '__main__':
    c = KNNTextClassifier()
    for k in range(1, 62, 2):
        c.setK(k)
        print "K = ", k
        runMultipleTests(c)
    for d in range(1, 3):
        c.setDistForm(d)
        print "distform =", d
        runMultipleTests(c)
    
