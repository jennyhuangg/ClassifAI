"""
Searches Twitter data for positive and negative dataset usernames.
"""

#-----------------------------------------------------------------------
# twitter-search
#  - performs a basic keyword search for tweets containing the keywords
#    "lazy" and "dog"
#-----------------------------------------------------------------------

from twitter import *

#-----------------------------------------------------------------------
# load our API credentials 
#-----------------------------------------------------------------------
config = {}
execfile("config.py", config)

#-----------------------------------------------------------------------
# create twitter API object
#-----------------------------------------------------------------------
twitter = Twitter(
                auth = OAuth(config["access_key"], config["access_secret"], config["consumer_key"], config["consumer_secret"]))

# Get usernames for positives
depression_phrases = [
    "I have depression",
    "I use antidepressants",
    "I was diagnosed with depression", 
    "I am depressed"
    ]

names = open('generatedata/names/positives_names', 'w')
for phrase in depression_phrases:
    
    #-----------------------------------------------------------------------
    # perform a basic search 
    # Twitter API docs:
    # https://dev.twitter.com/rest/reference/get/search/tweets
    #-----------------------------------------------------------------------
    query = twitter.search.tweets(q=phrase, result_type='recent', lang='en', count=1000)

    #-----------------------------------------------------------------------
    # How long did this query take?
    #-----------------------------------------------------------------------
    print "Search complete (%.3f seconds)" % (query["search_metadata"]["completed_in"])

    #-----------------------------------------------------------------------
    # Loop through each of the results, and print its content.
    #-----------------------------------------------------------------------

    for result in query["statuses"]:
        names.write("%s \n" % (result["user"]["screen_name"].encode('utf-8')))


# Get usernames for negatives
# http://techland.time.com/2009/06/08/the-500-most-frequently-used-words-on-twitter/
random_words = ['the', 'I', 'a']
neg_names = open('generatedata/names/negative_names', 'w')
for word in random_words:
    
    #-----------------------------------------------------------------------
    # perform a basic search 
    # Twitter API docs:
    # https://dev.twitter.com/rest/reference/get/search/tweets
    #-----------------------------------------------------------------------
    query = twitter.search.tweets(q=word,  result_type='recent', lang='en', count=77)

    #-----------------------------------------------------------------------
    # How long did this query take?
    #-----------------------------------------------------------------------
    print "Search complete (%.3f seconds)" % (query["search_metadata"]["completed_in"])

    #-----------------------------------------------------------------------
    # Loop through each of the results, and print its content.
    #-----------------------------------------------------------------------

    for result in query["statuses"]:
        neg_names.write("%s \n" % (result["user"]["screen_name"].encode('utf-8')))
