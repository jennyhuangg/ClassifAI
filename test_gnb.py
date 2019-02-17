"""
Implemented the Off-the-shelf algorithm that computes the Gaussian Naive Bayes 
Algorithm for generated continuous feature values.
Source: https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/
"""

import csv
import random, time
import math
 
def loadCsv(filename):
    lines = csv.reader(open(filename, "rb"))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset
 
def splitDataset(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet, copy]
 
def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated
 
def mean(numbers):
    return sum(numbers)/float(len(numbers))
 
def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
    return math.sqrt(variance)
 
def summarize(dataset):
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries
 
def summarizeByClass(dataset):
    separated = separateByClass(dataset)
    summaries = {}
    for classValue, instances in separated.iteritems():
        summaries[classValue] = summarize(instances)
    return summaries
 
def calculateProbability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent
 
def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.iteritems():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= calculateProbability(x, mean, stdev)
    return probabilities
            
def predict(summaries, inputVector):
    probabilities = calculateClassProbabilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.iteritems():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel
 
def getPredictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)
    return predictions
 
def getAccuracy(testSet, predictions):
    correct = 0
    for i in range(len(testSet)):
        if testSet[i][-1] == predictions[i]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0

def getPosAndNeg(predictions):
    pos_count = 0
    neg_count = 0
    for val in predictions:
        if val == 1.:
            pos_count += 1
        else:
            neg_count += 1
    print str(pos_count) + 'positive predictions'
    print str(neg_count) + 'negative predictions'
    print 'You predict a positive percentage of ' + str(float(pos_count) / (pos_count + neg_count))
 
def main():
    filename = 'generatedata/features/features.csv'
    splitRatio = 0.67
    dataset = loadCsv(filename)
    trainingSet, testSet = splitDataset(dataset, splitRatio)
    print('Split {0} rows into train={1} and test={2} rows').format(len(dataset), len(trainingSet), len(testSet))
    # prepare model
    summaries = summarizeByClass(trainingSet)
    # test model
    predictions = getPredictions(summaries, testSet)
    print predictions
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy: {0}%').format(accuracy)
    getPosAndNeg(predictions)
 

def multi_main(count):
    filename = 'generatedata/features/features.csv'
    splitRatio = 0.67
    dataset = loadCsv(filename)
    avg_acc = 0.
    avg_time = 0.
    for i in range(count):
        start_time = time.time()
        trainingSet, testSet = splitDataset(dataset, splitRatio)
        print('Split {0} rows into train={1} and test={2} rows').format(len(dataset), len(trainingSet), len(testSet))
        # prepare model
        summaries = summarizeByClass(trainingSet)
        # test model
        predictions = getPredictions(summaries, testSet)
        print predictions
        accuracy = getAccuracy(testSet, predictions)
        avg_acc += accuracy
        time_elapsed = time.time() - start_time
        avg_time += time_elapsed
        print("--- %s seconds ---" % (time_elapsed))
        print('Accuracy: {0}%').format(accuracy)
        getPosAndNeg(predictions)
    avg_acc = avg_acc / count
    print 'Average acc is ' +  str(avg_acc) + ' percent after ' + str(count) + ' attempts'
    print "Average runtime: ", avg_time / count

# main()
multi_main(50)