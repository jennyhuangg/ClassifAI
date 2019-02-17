"""
Contains commonly re-used helper functions for feature extraction.
"""

import string, re

def strip_non_ascii(string):
    ''' 
    Returns the string without non ASCII characters
    '''
    stripped = (c for c in string if 0 < ord(c) < 127)
    return ''.join(stripped)

def clean_tweet(tweet):
    '''
    Utility function to clean tweet text by removing links, special characters
    using simple regex statements.
    '''
    # http://www.geeksforgeeks.org/twitter-sentiment-analysis-using-python/
    tweet_text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())
    printable = set(string.printable)
    filter(lambda x: x in printable, tweet_text)
    tweet_text = strip_non_ascii(tweet_text).strip()
    tweet_text = tweet_text.replace('\n', ' ').replace('\r', '')
    return tweet_text

# given text, returns a list of n grams
# https://stackoverflow.com/questions/14617601/implementing-ngrams-in-python
def ngram(text, grams):  
    text = " ".join(text.split())
    text = format(text)
    words = text.split(" ")
    model = []  
    count = 0  
    for token in words[:len(words) - grams + 1]:  
       model.append(" ".join(words[count:count + grams]))  
       count = count + 1  
    return model
