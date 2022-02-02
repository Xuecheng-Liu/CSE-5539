try:
    import json
    import ast
except ImportError:
    import simplejson as json

tweets_filename = 'twitter_stream_200tweets.txt'
tweets_file = open(tweets_filename, "r",encoding='utf8')

count = 0;
languages = {}
percent = 0.0;



for line in tweets_file:
    try:
        tweet = ast.literal_eval(line)
        if tweet['country_code'] == "US":
            if tweet['lang'] not in languages:
                languages[tweet['lang']] == 1
            elif tweet['lang'] in languages:
                languages[tweet['lang']] += 1
            if tweet['geo'] != None:
                count += 1
    except:
        continue

if len(languages) != 0:
    for x in languages:
        percent = languages[x]*100.0/len(languages)
        print(f'{x} is {percent}%')
        print(f'{count*100/len(languages)} % tweets are geoTagged.')
elif len(languages) == 0:
    print(f'No tweets in US???')


