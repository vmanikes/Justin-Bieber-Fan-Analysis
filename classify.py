"""
classify.py
"""
import requests
import pickle as pkl
import re
import pandas as pd
from collections import Counter,defaultdict
from itertools import product
from scipy.sparse import lil_matrix
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from os import path
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
from sklearn.feature_extraction.text import CountVectorizer

def get_census_names():

    males = requests.get('http://www2.census.gov/topics/genealogy/1990surnames/dist.male.first').text.split('\n')
    females = requests.get('http://www2.census.gov/topics/genealogy/1990surnames/dist.female.first').text.split('\n')
    males_pct = dict([(m.split()[0].lower(), float(m.split()[1]))
                  for m in males if m])
    females_pct = dict([(f.split()[0].lower(), float(f.split()[1]))
                    for f in females if f])
    male_names = set([m for m in males_pct if m not in females_pct or
                  males_pct[m] > females_pct[m]])
    female_names = set([f for f in females_pct if f not in males_pct or
                  females_pct[f] > males_pct[f]])
    return male_names, female_names

def tokenize(string, keep_punctuation, prefix,
             collapse_urls, collapse_mentions):

    if not string:
        return []
    string = string.lower()
    tokens = []
    if collapse_urls:
        string = re.sub('http\S+', 'URL', string)
    if collapse_mentions:
        string = re.sub('@\S+', 'MENTION', string)
    if keep_punctuation:
        tokens = string.split()
    else:
        tokens = re.sub('\W+', ' ', string).split()
    if prefix:
        tokens = ['%s%s' % (prefix, t) for t in tokens]
    return tokens

def token_tweet(tweet, use_descr=True,
                 keep_punctuation=True, descr_prefix='d=',
                 collapse_urls=True, collapse_mentions=True):
    tokens = tokenize(tweet['text'], keep_punctuation, None,
                       collapse_urls, collapse_mentions)
    if use_descr:
        tokens.extend(tokenize(tweet['user']['description'],
                               keep_punctuation, descr_prefix,
                               collapse_urls, collapse_mentions))
    return tokens


def make_feature_matrix(tweets,tokens_list, vocabulary):
    X = lil_matrix((len(tweets), len(vocabulary)))
    for i, tokens in enumerate(tokens_list):
        for token in tokens:
            j = vocabulary[token]
            X[i,j] += 1
    return X.tocsr()

def find_best_setting(tweets,y, use_descr=True,
            keep_punctuation=True, descr_prefix=None,
            collapse_urls=True, collapse_mentions=True,):

    tokens = []
    for t in tweets:
        tokens.append(token_tweet(t, use_descr,
                                keep_punctuation, descr_prefix,
                                collapse_urls, collapse_mentions))
    vocab = defaultdict()
    features = []
    for i in tokens:
        for j in i:
            features.append(j)

    num_features = sorted(set(features))
    i = 0
    for j in num_features:
        vocab[j] = i
        i += 1
    X = lil_matrix((len(tweets), len(vocab)))
    for i, tokens1 in enumerate(tokens):
        for token in tokens1:
            j = vocab[token]
            X[i, j] += 1
    X = X.tocsr()
    cv = KFold(len(y), 5)
    accuracies = []
    for train_idx, test_idx in cv:
        clf = LogisticRegression()
        clf.fit(X[train_idx], y[train_idx])
        predicted = clf.predict(X[test_idx])

        acc = accuracy_score(y[test_idx], predicted)
        accuracies.append(acc)
    avg = np.mean(accuracies)

    return avg

def gender_by_name(tweets, male_names, female_names):
    for t in tweets:
        name = t['user']['name']
        if name:
            # remove punctuation.
            name_parts = re.findall('\w+', name.split()[0].lower())
            if len(name_parts) > 0:
                first = name_parts[0].lower()
                if first in male_names:
                    t['gender'] = 'male'
                elif first in female_names:
                    t['gender'] = 'female'
                else:
                    t['gender'] = 'unknown'
def tokenize_senti(text):
    return re.sub('\W+', ' ', text.lower()).split()

def afinn_sentiment2(terms, afinn):
    pos = 0
    neg = 0
    for t in terms:
        if t in afinn:
            if afinn[t] > 0:
                pos += afinn[t]
            else:
                neg += -1 * afinn[t]
    return pos, neg


def do_cross_validation(model, X, y, n_folds):
    cv = KFold(len(y), n_folds)
    accuracies = []
    for train_ind, test_ind in cv:
        model.fit(X[train_ind], y[train_ind])
        predictions = model.predict(X[test_ind])

        accuracies.append(accuracy_score(y[test_ind], predictions))
    print('Average 5-fold cross validation accuracy for the sentiment of tweets for Justin Bieber for positive and negative class=%.2f (std=%.2f)' %
            (np.mean(accuracies), np.std(accuracies)))

def main():
    male_names, female_names = get_census_names()
    pickle_in = open("tweets.pickle", "rb")
    tweets = pkl.load(pickle_in)

    data = []
    dummy = defaultdict(list)
    url = urlopen('http://www2.compute.dtu.dk/~faan/data/AFINN.zip')
    zipfile = ZipFile(BytesIO(url.read()))
    afinn_file = zipfile.open('AFINN/AFINN-111.txt')

    #CLASSIFYING ON SENTIMENT
    afinn = dict()

    for line in afinn_file:
        parts = line.strip().split()
        if len(parts) == 2:
            afinn[parts[0].decode("utf-8")] = int(parts[1])

    tweet_csv = pd.read_csv("Twitter.csv",delimiter=",")
    sentiment_labels = []

    for i in tweet_csv.itertuples():

        tokenize_tweet = tokenize_senti(i.tweets)
        pos,neg = afinn_sentiment2(tokenize_tweet,afinn)
        if pos > neg:
            dummy['positive'] = i.tweets
            sentiment_labels.append(1)
        elif neg > pos:
            dummy['negative'] = i.tweets
            sentiment_labels.append(-1)
        else:
            dummy['Neutral'] = i.tweets
            sentiment_labels.append(0)

    tweet_csv['senti_scores'] = sentiment_labels

    sentiment_labels = np.array(sentiment_labels)
    vectorizer = CountVectorizer()
    X_senti = vectorizer.fit_transform(tweet_csv['tweets'])

    model = LogisticRegression()
    do_cross_validation(model,X_senti,sentiment_labels,5)

    #END ON CLASSIFICATION BY SENTIMENT
    clean_tweets = []
    for t in tweets:
        if 'user' in t and 'name' in t['user']:
            parts = t['user']['name'].split()
            if len(parts) > 0:
                name = parts[0].lower()
        if name in male_names or name in female_names:
            clean_tweets.append(t)

    y = []
    for t in clean_tweets:
        if 'user' in t and 'name' in t['user']:
            parts = t['user']['name'].split()
            if len(parts) > 0:
                name = parts[0].lower()
        if name in female_names:
            dummy['female'] = t['text']
            dummy['female_name'] = t['user']['name']

            y.append(1)
        elif name in male_names:
            dummy['male'] = t['text']
            dummy['male_name'] = t['user']['name']
            y.append(0)
        else:
            y.append(-1)
    y = np.array(y)

    data.append(y)
    data.append(sentiment_labels)
    data.append(dummy)

    pickle_out = open("clean_tweets.pickle", "wb")
    pkl._dump(data, pickle_out)
    pickle_out.close()
    use_descr_opts = [True, False]
    keep_punctuation_opts = [True, False]
    descr_prefix_opts = ['d=', '']
    url_opts = [True, False]
    mention_opts = [True, False]

    argnames = ['use_descr', 'lower', 'punct', 'prefix', 'url', 'mention']
    option_iter = product(use_descr_opts,
                          keep_punctuation_opts,
                          descr_prefix_opts, url_opts,
                          mention_opts)
    results = []
    for options in option_iter:
        acc = find_best_setting(clean_tweets,y, *options)
        results.append((acc, options))

    final_acuracy = []
    for r in sorted(results, reverse=True):
        final_acuracy.append(('%.4f' % r[0], '  '.join('%s=%s' % (name, opt) for name, opt in zip(argnames, r[1]))))

    print("The top accuracy is ",final_acuracy[0][0])
    k = list(final_acuracy[0])
    print("For the setting",k[1:len(k)+1])

    for t in clean_tweets:
        t['gender'] = 'unknown'

    gender_by_name(clean_tweets, male_names, female_names)
    tweet_source = defaultdict(list)
    for i in clean_tweets:
        if i['gender'] == 'male':
            text = re.sub('<[^>]*>', '', i['source'])
            tweet_source['male'].append(text)
        else:
            text = re.sub('<[^>]*>', '', i['source'])
            tweet_source['female'].append(text)

    male_sources = Counter(tweet_source['male'])
    female_sources = Counter(tweet_source['female'])

    labels_male = [k for k,v in sorted(male_sources.items())]
    sizes_male = [v for k,v in sorted(male_sources.items())]
    plt.pie(sizes_male, labels=labels_male,
            autopct='%1.1f%%', shadow=True, startangle=140)
    plt.title("Tweets Sources of Male Users")
    plt.savefig("Male.pdf")
    plt.close()

    c = Counter(sentiment_labels)
    d = {}
    for key,values in c.items():
        if key == 1:
            d['positive'] = values
        elif key == -1:
            d['negative'] = values
        else:
            d['neutral'] = values

    labels_senti = [k for k,v in sorted(d.items())]
    values_senti = [v for k,v in sorted(d.items())]
    plt.pie(values_senti,labels=labels_senti,autopct='%1.1f%%', shadow=True, startangle=140)
    plt.title("Sentiment Analysis of Tweets")
    plt.savefig('sentiment.pdf')
    plt.close()
    labels_female = [k for k, v in sorted(female_sources.items())]
    sizes_female = [v for k, v in sorted(female_sources.items())]
    plt.pie(sizes_female, labels=labels_female,
            autopct='%1.1f%%', startangle=140)
    plt.title("Tweets Sources of Female Users")
    plt.savefig("Female.pdf")
    plt.close()

    k1 = k[1].split()
    target = open("WordList.txt",'w')
    word_list = [token_tweet(t, k1[0], k1[1],k1[2], k1[3],k1[4]) for t in tweets]
    for i in word_list:
        target.write(" ".join(i))
    target.close()

    d = path.dirname(__file__)

    # Read the whole text.
    text = open(path.join(d, 'WordList.txt')).read()
    word_cloud = WordCloud().generate(text)
    plt.imshow(word_cloud)
    plt.axis("off")

    # lower max_font_size
    wordcloud = WordCloud(max_font_size=40).generate(text)
    plt.figure()
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.savefig("WordCloud.pdf")
    plt.close()

    k = sorted(tweets,key = lambda x:x['created_at'])

if __name__ == '__main__':
    main()