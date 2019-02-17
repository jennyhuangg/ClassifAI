"""
Implements feature extraction from twitter object into matrix model in csv.
"""

import feature_helper
import numpy
import pickle
import csv
from textblob import TextBlob

#-----------------------------------------------------------------------
# Users have the following attributes (user_info, [tweets])
# - user_info has:
#       - followers: int
#       - friends: int
#       - lists: int
#       - favs: int
#       - statuses: int
# - each tweet has:
#       - time: string of UTC time
#       - text: string
#       - is_reply: 1 or 0
#       - is_rt: 1 or 0 (misleading name, says whether or not it was actually retweeted)
#       - favorites: int
#       - hashtags: int
#       - mentions: int
#       - utc-offset: int
#-----------------------------------------------------------------------

# Goes through a dictionary and adds on classification labels
def addLabels(tweet_dict, label):
    for tweet in tweet_dict.keys():
        tweet_dict[tweet] = tweet_dict[tweet] + (label,)

def getNumTweets(lst):
    return len(lst) 

def getAverageTweetLength(lst):
    len_count = 0.
    for t in lst:
        len_count += len(t['text'])
    return len_count / getNumTweets(lst)

# example UTC time: Sat Dec 02 19:47:24 +0000 2017
# gets the local time of a tweet given the user's utc-offset and 
# the utc posting time
def getLocalTime(tweet):
    try:
        offset = tweet['utc-offset'] / 3600
        # print tweet['utc-offset']
    except: 
        offset = 0
        # print 'blot'
    post_time = tweet['time'][11:19]
    post_hour = int(post_time[0:2])
    new_time = post_hour + offset
    if new_time < 0:
        new_time += 24
    # print (offset, post_time)
    # return str(new_time) + post_time[2:len(post_time)]
    return new_time

# Return the average number of favorites on a list of tweets
def getFavoriteAverage(lst):
    num_tweets = len(lst)
    count = 0.
    for tweet in lst:
        count += tweet['favorites']
    return count / num_tweets

# Given a user, get the number of tweets that are replies to other tweets
def getReplyRatio(lst):
    num_tweets = len(lst)
    count = 0.
    for tweet in lst:
        count += tweet['is_reply']
    return count / num_tweets

# sentiment analysis
def get_tweet_sentiment(tweet):
    analysis = TextBlob(clean_tweet(tweet))
    return  analysis.sentiment.polarity

# Get the ratio of positive to negative sentiment tweets
def getSentimentRatio(lst):
    pos_tweets = 1.
    neg_tweets = 1.
    for tweet in lst:
        temp_score = get_tweet_sentiment(tweet['text'])
        if temp_score < 0:
            neg_tweets += 1
        elif temp_score > 0:
            pos_tweets += 1
    return pos_tweets / (neg_tweets + pos_tweets)

# Given a list of a user's tweets, determine how many were tweeted late night 12am - 4am
def ratioInsomnia(lst):
    num_tweets = len(lst)
    count = 0.
    for tweet in lst:
        if getLocalTime(tweet) == 24 or getLocalTime(tweet) < 4:
            count += 1
    return count / num_tweets

# Gets the average number of mentions, returns as float
def getAvgMentions(lst):
    count = 0.
    num_tweets = len(lst)
    for tweet in lst:
        count += tweet['mentions']
    return count / num_tweets

# Get average sentiment
def getAvgSentiment(lst):
    count = 0.
    for tweet in lst:
        temp_score = get_tweet_sentiment(tweet['text'])
        count += temp_score
    return count / len(lst)

# Given a dictionary of user to tweets, increments all characteristics thats need to be increased by 1 for smoothing
def laplaceSmoothing(tweet_dict):
    # Iterate through every user
    for user in tweet_dict.keys():
        user_info = tweet_dict[user][0]
        user_tweets = tweet_dict[user][1]

        # Smooth all user info features
        user_info['followers'] += 1
        user_info['friends'] += 1
        user_info['lists'] += 1
        user_info['favs'] += 1
        user_info['statuses'] += 1

# Feature array will be [num tweets, num followers, num friends, follower friend ratio, average num favorites,
#                        reply rate, tweet sentiment ratio, label]
def generateTweetFeatures(tweet_dict):
    # Add our positive tweets to a CSV
    users = tweet_dict.keys()
    output = []

    for user in users:
        features = []

        user_attributes = tweet_dict[user]
        user_features = user_attributes[0]
        user_tweets = user_attributes[1]
        # Append num tweets
        # features.append(getNumTweets(user_tweets))

        # Append num followers
        features.append(user_features['followers'])

        # Append num friends
        # features.append(user_features['friends'])

        # Append follower friend ratio
        temp_ratio = float(user_features['followers']) / user_features['friends']
        features.append(temp_ratio)

        # Append average number of favorites
        avg_favs = getFavoriteAverage(user_tweets) / user_features['followers']
        features.append(avg_favs)

        # Append average reply rate
        # avg_reply = getReplyRatio(user_tweets)
        # features.append(avg_reply)

        # Append ratio of positive to negative tweets
        sentiment_ratio = getSentimentRatio(user_tweets)
        features.append(sentiment_ratio)

        # Get the average sentiment
        avg_sentiment = getAvgSentiment(user_tweets)
        features.append(avg_sentiment)

        # Get the ratio of tweets that are tweeted late into the night
        insomnia_ratio = ratioInsomnia(user_tweets)
        # features.append(insomnia_ratio)

        avg_mentions = getAvgMentions(user_tweets)
        features.append(avg_mentions)

        # Append the label
        features.append(tweet_dict[user][2])

        output += [features]

    return output

def main():
    # Load positives
    pkl_file = open('generatedata/timelines/positives.pkl', 'rb')
    positive_tweets = pickle.load(pkl_file)
    pkl_file.close()

    # Load control set
    pkl_file = open('generatedata/timelines/negatives.pkl', 'rb')
    negative_tweets = pickle.load(pkl_file)
    pkl_file.close()

    # Adds positive and negative scores indicating whether or not a user is depressed
    addLabels(positive_tweets, 1)
    addLabels(negative_tweets, 0)

    # print positive_tweets.keys()
    test_tweet = positive_tweets['kadusey2'][1][0]

    # Smooth both our dictionaries
    laplaceSmoothing(positive_tweets)
    laplaceSmoothing(negative_tweets)

    # generate features
    pos_features = generateTweetFeatures(positive_tweets)
    neg_features = generateTweetFeatures(negative_tweets)
    all_features = pos_features + neg_features

    with open("generatedata/features/features.csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerows(all_features)

main()