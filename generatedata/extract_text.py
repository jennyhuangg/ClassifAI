"""
Implements text extraction from twitter timelines into formatted text file.
"""

import pickle, random, string, re
import feature_helper

# Number of trials for cross validation
TRIALS = 10

# Phrases we will filter out
depression_phrases = [
    "depression",
    "antidepressants",
    "depressed"
    ]

def loadTweetDictionaries():
    '''
    Loads in dictionaries from pickle files
    '''
    # Load positives
    pkl_file = open('generatedata/timelines/positives.pkl', 'rb')
    positive_tweets = pickle.load(pkl_file)
    pkl_file.close()
    print len(positive_tweets)

    # Load control set
    pkl_file = open('generatedata/timelines/negatives.pkl', 'rb')
    negative_tweets = pickle.load(pkl_file)
    pkl_file.close()
    print len(negative_tweets)
    return positive_tweets, negative_tweets

# lst of users, and each user is [1, ['asdfd', 'asdfds',' asdfsd' ]]

def sortByUser(positive_tweets, negative_tweets):
    output = []
    pos_users = positive_tweets.keys()
    neg_users = negative_tweets.keys()
    # will be a tuple of (depressed or not, list of strings of tweets)

    for p in pos_users:
        tweet_lst = []
        user_tweets = positive_tweets[p][1]
        for t in user_tweets:
            tweet_lst += [feature_helper.clean_tweet(t['text'])]
        output += [(1, tweet_lst)]

    for n in neg_users:
        tweet_lst = []
        user_tweets = negative_tweets[n][1]
        for t in user_tweets:
            # print t['text'] + '\n'

            tweet_lst += [t['text']]
        output += [(0, tweet_lst)]

    print 'output'
    print len(output)

    # generate pickle vals of dictionary
    file = open('generatedata/timelines/user_tweet_list.pkl', 'wb')
    pickle.dump(output, file)

def readTweets(d, lst, label, ngrams):
    '''
    Process all tweets given a dict and put them into an array. Ngrams is 
    boolean denoting whether or not we'll be using multiple 
    '''
    users = d.keys()
    for user in users:
        user_tweets = d[user][1]
        tweets = ""
        for t in user_tweets:
            tweet_text = t['text']
            printable = set(string.printable)
            tweet_text = feature_helper.clean_tweet(tweet_text)
            hasPhrase = False
            for phrase in depression_phrases:
                temp_text = tweet_text
                if phrase in temp_text:
                    hasPhrase = True
            if not hasPhrase:
                tweets += " " + tweet_text
                if ngrams:
                    # random word that serves as marker denoting separate tweets
                    tweets += ' blumpy '
        lst += [str(label) + tweets]

def splitDataset(dataset, splitRatio):
    '''
    Given a list of n gram values, split into training and test set
    '''
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet, copy]

def writeToFile(filename, tweets):
    '''
    Given a list of tweets, write them out to a file
    '''
    with open(filename, 'w') as f:
        for user in tweets:
            # print user 
            f.write(user + '\n')

def createMultipleFiles(tweet_lst, ngrams):
    '''
    Creates a number of randomly sampled lists for cross validation
    '''
    for i in range(TRIALS):
        trainingSet, testSet = splitDataset(tweet_lst, 0.7)
        if ngrams:
            writeToFile('generatedata/text/training_ngram_tweets' + str(i), trainingSet)
            writeToFile('generatedata/text/test_ngram_tweets' + str(i), testSet)
        else:
            writeToFile('generatedata/text/training_tweets' + str(i), trainingSet)
            writeToFile('generatedata/text/test_tweets' + str(i), testSet)


def main():
    positive_tweets, negative_tweets = loadTweetDictionaries()
    sortByUser(positive_tweets, negative_tweets)
    list_of_tweets = []
    readTweets(positive_tweets, list_of_tweets, 1, False)
    readTweets(negative_tweets, list_of_tweets, 0, False)
    createMultipleFiles(list_of_tweets, False)

    ngram_list_of_tweets = []
    readTweets(positive_tweets, ngram_list_of_tweets, 1, True)
    readTweets(negative_tweets, ngram_list_of_tweets, 0, True)
    createMultipleFiles(ngram_list_of_tweets, True)

main()
