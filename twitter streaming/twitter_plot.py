# Import the necessary package to process data in JSON format
try:
    import json
    import ast
    import langid
    import matplotlib.pyplot as plt
    import pandas as pd
    import operator
except ImportError:
    import simplejson as json


tweets_filename = 'twitter_stream_200tweets.txt'
tweets_file = open(tweets_filename, "r",encoding='utf8')

language_dic = {}


for line in tweets_file:
    try:
        tweet = ast.literal_eval(line)
        if tweet['lang'] in language_dic:
            language_dic[tweet['lang']] += 1
        elif tweet['lang'] not in language_dic:
            language_dic[tweet['lang']] = 1
    except:
        continue

sorted_dic = dict(sorted(language_dic.items(),key = operator.itemgetter(0),reverse = True))


plt.bar(*zip(*sorted_dic.items()))
plt.show()

