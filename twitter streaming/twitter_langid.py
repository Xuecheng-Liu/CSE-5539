try:
    import json
    import ast
    import langid
except ImportError:
    import simplejson as json

tweets_filename = 'twitter_stream_200tweets.txt'
tweets_file = open(tweets_filename, "r",encoding='utf8')

agreed = []
disagreed = []

for line in tweets_file:
    try: 
        tweet = ast.literal_eval(line)
        if 'text' in tweet :
            if langid.classify('text') == langid.classify(tweet['lang']):
                if tweet['lang'] not in agreed:
                    agreed.append(tweet['lang'])
            elif langid.classify('text') != langid.classify(tweet['lang']):
                if tweet['lang'] not in disagreed:
                    disagreed.append(tweet['lang'])
    except:
        continue

print(f'Number of tagged languages are: {len(agreed)+len(disagreed)}')

if len(disagreed) != 0:
    print(f'langid agree with twitter for {len(agreed)*100.0/len(disagreed)}%')
    for langs in disagreed:
        print(f'These languages are disagrred by langid: {langs}')
elif len(disagreed) == 0:
    print(f'ALL GOOD')
    


