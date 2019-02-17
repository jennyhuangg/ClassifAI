"""
Queries Twitter timelines given usernames, outputs to pkl files.
"""

#-----------------------------------------------------------------------
# twitter-search
#  - performs a basic keyword search for tweets containing the keywords
#    "lazy" and "dog"
#-----------------------------------------------------------------------

from twitter import *
import pickle

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

# EXTRACT FEATURES FROM USERS AND TIMELINES

#-----------------------------------------------------------------------
# access list of user names
#-----------------------------------------------------------------------

name_files = ["generatedata/names/negative_names", "generatedata/names/positive_names"]
data_files = ["generatedata/timelines/negatives.pkl", "generatedata/timelines/positives.pkl"]

for i in range(2):
    with open(name_files[i]) as f:
        content = f.readlines()
    content = [x.strip() for x in content] 

    def updateUserDict(d, a1, a2, a3, a4, a5):
        d['followers'] = a1
        d['friends'] = a2
        d['lists'] = a3
        d['favs'] = a4
        d['statuses'] = a5
        return d

    print len(content)
    # creates a user dictionary with the key being the username, and value will be a list of tweets
    tweet_dict = {}

    # get statuses based on names
    count = 0
    for name in content:
        count += 1
        print count
        # gets at most 1000 tweets from each user timeline
        try:
            query = twitter.statuses.user_timeline(screen_name=name, count=1000)
        except:
            continue
        
        if len(query) == 0:
            continue

        # get the information for a single user
        user_info = query[0]['user']
        user_followers = user_info['followers_count']
        user_friends = user_info['friends_count']
        user_lists = user_info['listed_count']
        user_favs = user_info['favourites_count']
        user_statuses = user_info['statuses_count']
        # Creates a dictionary of values
        user_dict = updateUserDict({}, user_followers, user_friends, user_lists, user_favs, user_statuses)

        tweet_list = []
        # for every tweet in the query
        for result in query:
            tweet_time = result['created_at']
            
            tweet_text = result['text']
            
            # if the tweet is replying to another, tweet_is_reply = 1, else 0
            if result['in_reply_to_status_id']:
                tweet_is_reply = 1
            else:
                tweet_is_reply = 0
            
            tweet_time_offset = result['user']['utc_offset']
            tweet_is_retweet = int(result['retweeted'] == True)
            tweet_favorites = result['favorite_count']
            tweet_num_hashtags = len(result['entities']['hashtags'])
            tweet_mentions = len(result['entities']['user_mentions'])

            # creates a dictionary of attributes
            tweet_attributes = {}
            tweet_attributes['time'] = tweet_time
            tweet_attributes['text'] = tweet_text
            tweet_attributes['is_reply'] = tweet_is_reply
            tweet_attributes['is_rt'] = tweet_is_retweet
            tweet_attributes['favorites'] = tweet_favorites
            tweet_attributes['hashtags'] = tweet_num_hashtags
            tweet_attributes['mentions'] = tweet_mentions
            tweet_attributes['utc-offset'] = tweet_time_offset
            tweet_list += [tweet_attributes]

        tweet_dict[name] = (user_dict, tweet_list)

    # generate pickle vals of dictionary
    output = open(data_files[i], 'wb')
    pickle.dump(tweet_dict, output)
