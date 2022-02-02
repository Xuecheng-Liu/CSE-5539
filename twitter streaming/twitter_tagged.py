# Import the necessary package to process data in JSON format
try:
    import json
    import ast
except ImportError:
    import simplejson as json

# We use the file saved from last step as example
tweets_filename = 'twitter_stream_200tweets.txt'
tweets_file = open(tweets_filename, "r",encoding='utf8')

count_total = 0;
count_tagged = 0;

for line in tweets_file:
    try:
        count_total += 1
        # Read in one line of the file, convert it into a json object 
        tweet = ast.literal_eval(line)
        if 'lang' in tweet :
            count_tagged += 1
    except:
        # read in a line is not in JSON format (sometimes error occured)
        continue
    
print(f'Total is: {count_total}')
print(f'LangID Tagged tweets is: {count_tagged}')


if count_tagged < count_total:
    print(f'{(count_tagged/count_total)*100.0} % tweets are tagged.')
