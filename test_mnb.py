"""
Based off the Bag-of-words algorithm implemented in HW5, we have created a Naive Bayes 
algorithm that calculates the probability of every word occurring in a depressed individual's
tweet, and use these probabilities to make an overall prediction.
"""

from math import log, fabs
import sys, pickle, random, copy, re, string, time
import generatedata.feature_helper as feature_helper

# Number of trials
TRIALS = 10

# THRESHOLD = 0.58, best is .58

class TweetClassifier:
    def runNGramTraining(self, infile, num_grams):
        self.dict = {}
        self.counts = [[], []]
        self.nrated = [0,0]

        # with open(infile) as f:
        #     content = f.readlines()
        # print len(content)
        # content = feature_helper.ngram(content, num_grams)
        # # remove grams that span between tweets
        # for x in content:
        #     if 'blumpy' in x:
        #         content.remove(x)
        with open(infile) as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        index_count = 0
        for review in content:
            rating = int(review[0])
            review = review[1:]
            temp_lst = feature_helper.ngram(review, num_grams)
            for x in temp_lst:
                if 'blumpy' in x:
                    temp_lst.remove(x)
            # print temp_lst
            
            # increase the count for a review of a given rating
            self.nrated[rating] += 1
            for i in range(1, len(temp_lst)):
                word = temp_lst[i]
                # check if the word is not yet in the dictionary
                if word not in self.dict:
                    self.dict[word] = index_count
                    index_count += 1
                    # increase the length of each array by 1
                    for array in self.counts:
                        array += [0]
                # else increase the value by one
                self.counts[rating][self.dict[word]] += 1
        # print self.dict


    def runTraining(self, infile):
        """
        Builds a dictionary based off the inputted training set, and counts the occurrences of words
        in each category
        """
        self.dict = {}
        self.counts = [[], []]
        self.nrated = [0,0]

        with open(infile) as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        index_count = 0
        for review in content:
            temp_lst = review.split()
            # print temp_lst
            rating = int(temp_lst[0])
            # increase the count for a review of a given rating
            self.nrated[rating] += 1
            for i in range(1, len(temp_lst)):
                word = temp_lst[i]
                # check if the word is not yet in the dictionary
                if word not in self.dict:
                    self.dict[word] = index_count
                    index_count += 1
                    # increase the length of each array by 1
                    for array in self.counts:
                        array += [0]
                # else increase the value by one
                self.counts[rating][self.dict[word]] += 1
        # print self.dict

    def fitModel(self, alpha=1):
        """
        Try fitting the model now with a given alpha
        """
        num_words = len(self.counts[0])
        self.F = [[],[]]

        for i in range(2):
            count_lst = self.counts[i]
            rating_sum = 0
            for x in count_lst:
                rating_sum += x + alpha

            for j in range(num_words):
                self.F[i] += [-log(float(count_lst[j] + alpha) / rating_sum)]
               

    def predictSingleTweet(self, tweet):
        """
        Given a single tweet, use our model and predict whether or not it is a depressed individual
        """
        tweet = tweet.split()
        # print 'tweet is blumpy'
        # print tweet

        rating_probabilities = []
        # iterate through every possible rating
        for i in range(2):
            # iterate through every word in a line
            temp_prob = -log(self.nrated[i] / float(sum(self.nrated)))
            for j in range(1, len(tweet)):
                try:
                    temp_prob += self.F[i][self.dict[tweet[j]]]
                except:
                    continue
            rating_probabilities += [temp_prob]
        prediction = rating_probabilities.index(min(rating_probabilities))
        return prediction

    def predictSingleUser(self, user_info, THRESHOLD):
        if len(user_info) == 0:
            print 'empty boi'
            return 0
        tweets = user_info
        # rating = user_info[0]
        count = 0.
        num_tweets = 0
        for t in tweets:
            if len(t) == 0:
                continue
            num_tweets += 1
            temp_prediction = self.predictSingleTweet(t)
            count += temp_prediction
        overall = (count + 1) / (num_tweets + 1)
        if overall > THRESHOLD:
            return 1
        else:
            return 0

    def splitDataset(self, dataset, splitRatio):
        '''
        Given a list of n gram values, split into training and test set
        '''
        print len(dataset), 'is the dataset length'
        trainSize = int(len(dataset) * splitRatio)
        trainSet = []
        c = copy.copy(dataset)
        print len(c)
        print 'copy length'
        while len(trainSet) < trainSize:
            index = random.randrange(len(c))
            trainSet.append(c.pop(index))
        # print trainSet[0]
        return [trainSet, c]

    def createFileFromList(self, dataset, filename):
        # print dataset
        print len(dataset), 'is da length'
        with open(filename, 'w') as f:
            for user in dataset:
                try: 
                    # print 'user:'
                    # print user
                    temp_string = str(user[0])
                    for tweet in user[1]:
                        temp_string += ' ' + feature_helper.clean_tweet(tweet)
                    f.write(temp_string + '\n')
                except:
                    continue

    def testModel(self, infile):
        """
        Test the accuracy of the model we generated
        """
        with open(infile) as f:
            content = f.readlines()
        content = [x.strip() for x in content]

        num_correct = 0
        num_reviews = len(content)
        outputs = []
        c_matrix = [[0, 0], [0, 0]]


        for tweet in content:
            rating = int(tweet[0])
            prediction = self.predictSingleTweet(tweet)
            outputs += [prediction]
            if prediction == rating:
                if prediction == 1:
                    c_matrix[0][0] += 1
                else:
                    c_matrix[1][1] += 1
                num_correct += 1
            else:
                if prediction == 1:
                    c_matrix[0][1] += 1
                else:
                    c_matrix[1][0] += 1
        return (outputs, num_correct/ float(num_reviews), c_matrix)

    def predictNGramTweet(self, tweet, num_grams):
        """
        Given a single tweet, use our model and predict whether or not it is a depressed individual
        """
        tweet = feature_helper.ngram(tweet, num_grams)
        for x in tweet:
            if 'blumpy' in x:
                tweet.remove(x)
 
        rating_probabilities = []
        # iterate through every possible rating
        for i in range(2):
            # iterate through every word in a line
            temp_prob = -log(self.nrated[i] / float(sum(self.nrated)))
            for j in range(1, len(tweet)):
                try:
                    temp_prob += self.F[i][self.dict[tweet[j]]]
                except:
                    continue
            rating_probabilities += [temp_prob]
        prediction = rating_probabilities.index(min(rating_probabilities))
        return prediction

    def predictBothTweet(self, text, num_grams):
        """
        Given a single tweet, use our model and predict whether or not it is a depressed individual
        """
        words = text.split()
        tweet = feature_helper.ngram(text, num_grams)
        for x in tweet:
            if 'blumpy' in x:
                tweet.remove(x)
        # removes all blumpies in a list of words
        words[:] = [x for x in words if x != 'blumpy'] 
 
        rating_probabilities = []
        # iterate through every possible rating
        for i in range(2):
            # iterate through every word in a line
            temp_prob = -log(self.nrated[i] / float(sum(self.nrated)))
            for j in range(1, len(tweet)):
                try:
                    temp_prob += self.F[i][self.dict[tweet[j]]]
                except:
                    continue
            for j in range(1, len(words)):
                try:
                    temp_prob += self.F[i][self.dict[words[j]]]
                except:
                    continue
            rating_probabilities += [temp_prob]
        prediction = rating_probabilities.index(min(rating_probabilities))
        return prediction


    def testNGramModel(self, infile, num_grams):
        """
        Test the accuracy of the model we generated
        """
        with open(infile) as f:
            content = f.readlines()
        content = [x.strip() for x in content]

        num_correct = 0
        num_reviews = len(content)
        outputs = []
        c_matrix = [[0, 0], [0, 0]]

        for tweet in content:
            rating = int(tweet[0])
            prediction = self.predictNGramTweet(tweet, num_grams)
            outputs += [prediction]
            if prediction == rating:
                if prediction == 1:
                    c_matrix[0][0] += 1
                else:
                    c_matrix[1][1] += 1
                num_correct += 1
            else:
                if prediction == 1:
                    c_matrix[0][1] += 1
                else:
                    c_matrix[1][0] += 1

        return (outputs, num_correct/ float(num_reviews), c_matrix)
   
    def testNGramsAndBag(self, infile, num_grams):
        with open(infile) as f:
            content = f.readlines()
        content = [x.strip() for x in content]

        num_correct = 0
        num_reviews = len(content)
        outputs = []
        c_matrix = [[0, 0], [0, 0]]

        for tweet in content:
            rating = int(tweet[0])
            tweet = tweet[1:]
            prediction = self.predictBothTweet(tweet, num_grams)
            outputs += [prediction]
            if prediction == rating:
                if prediction == 1:
                    c_matrix[0][0] += 1
                else:
                    c_matrix[1][1] += 1
                num_correct += 1
            else:
                if prediction == 1:
                    c_matrix[0][1] += 1
                else:
                    c_matrix[1][0] += 1

        return (outputs, num_correct/ float(num_reviews), c_matrix)

    def alphaOptimization(self, infile):
        """
        Test various values of alpha and determine the best one
        """
        best_alpha = 0.1
        temp_alpha = 0.1
        best_accuracy = 0
        while temp_alpha < 5:
            self.fitModel(temp_alpha)
            temp_acc = self.testModel(infile)[1]
            if temp_acc > best_accuracy:
                best_accuracy = temp_acc
                best_alpha = temp_alpha
            temp_alpha += .1
        return best_alpha

def runByUser(THRESHOLD):
    pkl_file = open('generatedata/timelines/user_tweet_list.pkl', 'rb')
    user_tweet_list = pickle.load(pkl_file)
    pkl_file.close()
    c = TweetClassifier()

    train, test = c.splitDataset(user_tweet_list, 0.7)
    c.createFileFromList(train, 'generatedata/features/user_test0')
    c.runTraining('generatedata/features/user_test0')
    # print len(c.dict), "words in dictionary"
    # print "Fitting model..."
    c.fitModel(0.01)

    correct_guesses = 0
    for t in test:
        real_rating = t[0]
        tweet_lst = t[1] 
        prediction = c.predictSingleUser(tweet_lst, THRESHOLD)
        if prediction == real_rating:
            correct_guesses += 1
    print 'Your accuracy by user is ', correct_guesses / float(len(test))
    return correct_guesses / float(len(test))

def runMultipleTimesByUser(x, THRESHOLD):
    avg = 0.
    for i in range(x):
        avg += runByUser(THRESHOLD)
    return avg / x

def findBestThreshold():
    i = .58
    best_avg = 0.
    best_threshold = .6
    while i < .62:
        avg = runMultipleTimesByUser(30, i)
        if avg > best_avg:
            best_avg = avg
            best_threshold = i
        i += .05
    return avg, best_threshold

# tewsting for bag of words model
def runMultipleTests(c):
    '''
    Performs testing on the different files
    '''
    avg_acc = 0.
    avg_sens = 0.
    avg_spec = 0.
    avg_prec = 0.
    avg_time = 0.
    for i in range(TRIALS):
        start_time = time.time()
        c.runTraining('generatedata/text/training_tweets' + str(i))
        print len(c.dict), "words in dictionary"
        print "Fitting model..."
        c.fitModel(.001)
        #temp_result = c.testNGramsAndBag('generatedata/text/test_ngram_tweets' + str(i), 5)
        temp_result = c.testModel('generatedata/text/test_tweets' + str(i))
        temp_acc = temp_result[1]
        cm = temp_result[2]
        print "Accuracy on validation set:", temp_acc
        print "Confusion matrix: ", cm
        true_pos = cm[0][0]
        false_pos = cm[0][1]
        false_neg = cm[1][0]
        true_neg = cm[1][1]
        avg_spec += float(true_neg) / (true_neg + false_pos)
        avg_sens += float(true_pos) / (true_pos + false_neg)
        avg_prec += float(true_pos) / (true_pos + false_pos)
        avg_acc += temp_acc
        time_elapsed = time.time() - start_time
        avg_time += time_elapsed
        print("--- %s seconds ---" % (time_elapsed))
    print "\nAverage overall accuracy:", avg_acc / TRIALS
    # print "Final cm:", true_pos, false_pos, '\n', false_neg, true_neg  
    print "Average runtime: ", avg_time / TRIALS
    print "Average specificity: ", avg_spec / TRIALS
    print "Average sensitivity: ", avg_sens / TRIALS
    print "Average precision: ", avg_prec / TRIALS

# testing for n gram model
def runNGramPipeline(c, num_grams):
    accuracy = 0.
    runtime = 0.
    avg_sens = 0.
    avg_spec = 0.
    avg_prec = 0.
    for i in range(TRIALS):
        start_time = time.time()
        c.runNGramTraining('generatedata/text/training_ngram_tweets' + str(i), num_grams)
        c.fitModel()
        a = c.testNGramModel('generatedata/text/test_ngram_tweets' + str(i), num_grams)
        cm = a[2]
        print "Confusion matrix: ", cm
        true_pos = cm[0][0]
        false_pos = cm[0][1]
        false_neg = cm[1][0]
        true_neg = cm[1][1]
        avg_spec += float(true_neg) / (true_neg + false_pos)
        avg_sens += float(true_pos) / (true_pos + false_neg)
        avg_prec += float(true_pos) / (true_pos + false_pos)
        accuracy += a[1]
        runtime += (time.time() - start_time)
    print "Overall Average Accuracy: ", accuracy / TRIALS
    print "Average runtime:", runtime / TRIALS
    print "Average specificity: ", avg_spec / TRIALS
    print "Average sensitivity: ", avg_sens / TRIALS
    print "Average precision: ", avg_prec / TRIALS

# testing for ngram and bag of words model
def runGramAndBag(c):
    num_grams = 5
    for i in range(TRIALS):
        c.runTraining('generatedata/text/training_tweets' + str(i))
        print len(c.dict), 'phrases/words in the dictionary'
        c.fitModel()
        print c.testNGramsAndBag('generatedata/text/test_ngram_tweets' + str(i), num_grams)
        print c.testModel('generatedata/text/test_tweets' + str(i))

if __name__ == '__main__':
    c = TweetClassifier()
    runMultipleTests(c)
    # runByUser(.58)
    for n in range(1, 9):
        print "N =", n
        runNGramPipeline(c, n)
    #runNGramPipeline(c, 5)
    # runGramAndBag(c)
    
