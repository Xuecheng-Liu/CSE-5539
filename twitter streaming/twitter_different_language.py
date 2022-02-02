# Import the necessary package to process data in JSON format
try:
    import json
    import ast
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

print(f'There are in total {len(language_dic)} different languages')
total = 0;
for lang in language_dic:
    total += language_dic[lang]

for lang in language_dic:
    percent = language_dic[lang]*100.0/total
    print(f'{lang} is {percent} %')




