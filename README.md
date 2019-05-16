import nltk
from nltk.corpus import stopwords
import string
import gensim
from gensim import corpora
import tweepy

tweetList=[]
stops = set(stopwords.words("english"))
stemmer = nltk.stem.SnowballStemmer('english')
lemmatizer = nltk.wordnet.WordNetLemmatizer()

# Twitter API credentials
consumer_api_key = ""
consumer_api_secret_key = ""
access_token = ""
access_token_secret = ""

def get_all_tweets(screen_name):
    # Twitter only allows access to a users most recent tweets with this method
    # authorize twitter, initialize tweepy
    auth = tweepy.OAuthHandler(consumer_api_key, consumer_api_secret_key)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)

    # initialize a list to hold all the tweepy Tweets
    alltweets = []

    # make initial request for most recent tweets
    new_tweets = api.user_timeline(screen_name=screen_name, count=400)

    # save most recent tweets
    alltweets.extend(new_tweets)

    # save the id of the oldest tweet less one
    oldest = alltweets[-1].id - 1

    # keep grabbing tweets until there are no tweets left to grab
    while len(new_tweets) > 195:
        print("getting tweets before %s" % (oldest))

        # all subsiquent requests use the max_id param to prevent duplicates
        new_tweets = api.user_timeline(screen_name=screen_name, count=200, max_id=oldest)

        # save most recent tweets
        alltweets.extend(new_tweets)

        # update the id of the oldest tweet less one
        oldest = alltweets[-1].id - 1

        print("...%s tweets downloaded so far" % (len(alltweets)))

    for tweet in alltweets:
        rawTweet = tweet._json['text']
        print(rawTweet)
        processedTweet = rawTweet.strip()
        processedTweet = processedTweet.translate(str.maketrans('','',string.punctuation))
        tweetTokens = nltk.word_tokenize(processedTweet)

        for token in tweetTokens:
            if token in stops:
                tweetTokens.remove(token)

        for token in tweetTokens:
            oldToken = token
            tweetTokens.remove(token)
            oldToken = lemmatizer.lemmatize(oldToken)
            tweetTokens.append(oldToken)

        processedTweet = ' '.join(tweetTokens)
        print(processedTweet)

        if len(tweetTokens) > 3:
            tweetList.append(processedTweet)
    #print(tweetList)

    texts = [[text for text in doc.split()] for doc in tweetList]
    print(texts)
    dictionary = corpora.Dictionary(texts)
    print("printing dictionary",dictionary.token2id)
    print(dictionary)
    doc_term_matrix = [dictionary.doc2bow(doc.split()) for doc in tweetList]
    #print(doc_term_matrix)
    ldaObject = gensim.models.ldamodel.LdaModel
    ldaModel = ldaObject(doc_term_matrix,num_topics=3,id2word=dictionary,passes=20)
    print(ldaModel.print_topics(num_topics=3, num_words=10))
    print("LDA analysis complete")

    pass


if __name__ == '__main__':
    # pass the username of the account you want to download tweets
    get_all_tweets("@NIUlive")
