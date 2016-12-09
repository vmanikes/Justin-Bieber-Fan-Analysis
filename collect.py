"""
collect.py
"""
from TwitterAPI import TwitterAPI
import configparser
import pandas as pd
import re
import sys
import time
import pickle as pkl

def get_twitter(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)
    twitter = TwitterAPI(
                   config.get('twitter', 'consumer_key'),
                   config.get('twitter', 'consumer_secret'),
                   config.get('twitter', 'access_token'),
                   config.get('twitter', 'access_token_secret'))
    return twitter

def robust_request(twitter, resource, params, max_tries=5):

    for i in range(max_tries):
        request = twitter.request(resource, params)
        if request.status_code == 200:
            return request
        else:
            print('Got error %s \nsleeping for 15 minutes.' % request.text)
            sys.stderr.flush()
            time.sleep(61 * 15)


def main():

    tweets = []
    twitter = get_twitter('cred1.cfg')
    request = robust_request(twitter, 'search/tweets', {'q': '@justinbieber', 'result_type': 'mixed', 'count': 100,'lang':'en','geo':'41.8781,87.6298,20mi'})
    for i in request:
        tweets.append(i)
        if len(tweets) % 100 == 0:
            print('%d tweets' % len(tweets))

    tweet_id = tweets[-1]['id']

    for i in range(0,9):
        request = robust_request(twitter, 'search/tweets', {'q': '@justinbieber', 'result_type': 'mixed', 'count': 100, 'max_id':tweet_id,'lang':'en','geo':'41.8781,87.6298,20mi'})
        for i in request:
            if i not in tweets:
                tweets.append(i)
            if len(tweets) % 100 == 0:
                print('%d tweets' % len(tweets))

        tweet_id = tweets[-1]['id']

    pickle_out = open("tweets.pickle","wb")
    pkl._dump(tweets,pickle_out)
    pickle_out.close()

    graph_file = open('GraphList.txt', 'w')

    screen_name = []

    for i in tweets:
        screen_name.append(i['user']['screen_name'])

    screen_name = list(set(screen_name))

    user_ids = []
    for i in screen_name[0:20]:
        request = robust_request(twitter, 'followers/ids', {'screen_name': i, 'count': 200})
        for j in request:
            user_ids.append(j)
            graph_file.write("%s\t%s\n" % (i, str(j)))

    id = []
    tweet = []
    tweet_source = []
    screen_name = []
    friends_count = []
    follower_count = []
    user_id = []
    name = []

    for i in tweets:
        if i['id'] not in id:
            id.append(i['id'])
            name.append(i['user']['name'])
            #description.append(i['user']['description'])
            user_id.append(i['user']['id_str'])
            tweet.append(i['text'])
            source = re.sub('<[^>]*>', '', i['source'])
            tweet_source.append(source)
            screen_name.append(i['user']['screen_name'])
            friends_count.append(i['user']['friends_count'])
            follower_count.append(i['user']['followers_count'])

    df = pd.DataFrame()
    df['id'] = id
    df['user_id'] = user_id
    df['name'] = name
    df['source_of_tweet'] = tweet_source
    df['screen_name'] = screen_name
    df['friends'] = friends_count
    df['followers'] = follower_count
    df['tweets'] = tweet

    df.to_csv('Twitter.csv')

if __name__ == '__main__':
    main()