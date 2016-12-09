# Justin-Bieber-Fan-Analysis

In this Assignment I have determined the number of female and male tweets of Justin Bieber in Chicago, sentiment analysis of the
tweets and how will the people will tweet i.e using an Iphone, Android or some other mode. I have done clustering of the
users that tweet about justin Bieber using the Girvan Newman Algorithm. The detailed explaination for each of these is in
the following paragraphs.

Data collection:
----------------
I have used the Twitter API to collect the data for the tweets that mention Justin Bieber which come from Chicago and
those tweets that are in english. I stored all the tweets in a "pickle" and in a CSV file where the columns in the CSV
file include:
    TweetId
    Name
    UserId
    Source of Tweet
    Screen Name
    friends
    followers
    tweets

I have done the data collection in two steps where in the first step we would get the 100 tweets and in the next query
if we want to get the tweets from 101, we need to specify in the next query as "max_id" parameter where the value for this
would be the "tweetId" of the 100th tweet. But the problem with this is that when we specify the max_id parameter we would
get the 100th tweet also which will result in repetetion of tweets. So to get the required number of tweets we need to do
the following:
    Change the range() in the for loop to tweak with the nuber of tweets that you want to test with. For example, if the range
    (0,9) we will get 100 (First Query) + 900 ( from the loop) = 991 tweets ( removing duplicates).
This is what is done in the Data collection Part

Data Classification:
--------------------
In the data classification part I have done 2 types of classification:
    a) By Sentiment
    b) By Gender

    By Sentiment
    ------------
    In this I have taken the "tweets" that were stored in the CSV file using the "Pandas" Library. My friend for this part
    of the classification is the AFINN word set which is used to classify the tweets negative or positive. The graph for this
    part is labelled as "sentiment.pdf". So What I have done in this module is that:

        1) I have taken the tweets
        2) Tokenized them
        3) Computed the AFINN Scores for the positive and negative words, i.e., if the Afinn Score of the word is greater than 0
           I add +1 for the positive score else I add +1 for the negative score
        4) After getting the positive and negative scores if the positive score is greater than negative score I add 1 as the
           label for the tweet, if negative score is higher I add -1 to the label and if both are equal I add 0 to the label
        5) I also vectorize all the tweets by the "sklearn's CountVectorizer"
        6) So I now have the features and labels, Now I will build a LogisticRegression classifier and I will fit the features and
           labels to the classifier and done the "CrossValidation" using 5 folds.

    By Gender
    ---------
    For this part of the classification my friend is the US census data where I get the names of the people from the US
    categorized by the frequency of the words. I have loaded the pickle file that we have saved in the "Data collection" part.
    I have iterated over the tweets and took onlt the tweets that were matching with the census data. This is done because
    to clear the noise in the data. After doing that I tokenized the tweets using different modes such as keep_mentions,
    collapse url, keep punctuations. I did this and built a feature matrix. For the labels I have iterated over the tweets
    which were fi ltered, if the name appears in female add 1 to the labels else 0 to the male. I have done this by permuting
    on the modes by fitting a LogisticRegression classifier as above and running it for different settings and picking out the
    best setting that gives the max accuracy


Cool things done in classification:
----------------------------------
As I get the female and male tweets classified,I have a feature in each of the tweets which tells me how the person has tweeted
the tweet, i.e., by iphone, Android, Mac or somethig else. I felt that this data would be useful for companies like apple to
to see how many males are using iphone to tweet and how many female are using iphone to tweet and how they can improve.
Twitter also can use this I have saved the files of the
pie charts as "Male.pdf" and "Female.pdf"

Another feature is that For the best setting I have created a word cloud image which will show what words have appeared in a
interactive way rather than a boring old print statement. I have saved the image as "WordCloud.pdf"

Clustering:
----------
In the cluster.py I have taken all the users that I have collected the list of people who have mentioned justin bieber in
their tweets, after this I have taken a subset of users from the list of users that I have collected ( let's say 20 users
because of the rate limiting issues of twitter) I then collected the followes of each of the 20 users.The graphs before
making the clusters is stored in 'cluster.pdf', For each user I have collected 200 followers. I have drawn the network
like this, for example (-> indicates follower)
        A1 -> A
        A2 -> A
        A3 -> A
        B1 -> B
        B2 -> B
        A1 -> B

    The network would be like
            A
          / | \
        A1  A2  A3
        |
        B
      /   \
     B1   B2

I made a network like this and applied the Networkx's Girvan Newmann method to make the graph into clusters
