"""
sumarize.py
"""
import pickle
import re

pickle_in = open("Cluster_pickle.pickle","rb")
clusters = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open("tweets.pickle","rb")
tweets = pickle.load(pickle_in)
pickle_in.close()


users_count = []
for i in tweets:
    users_count.append(i['user']['id'])

users_count = set(users_count)
msg_count = []
for i in tweets:
    if i['text']:
        msg_count.append(i['text'])

pickle_in = open("clean_tweets.pickle","rb")
classes = pickle.load(pickle_in)
pickle_in.close()

male = 0
female = 0
for i in classes[0]:
    if i == 1:
        female +=1
    else:
        male +=1

pos = 0
neg = 0
neu = 0
for i in classes[1]:
    if i == 1:
        pos += 1
    elif i == -1:
        neg += 1
    else:
        neu += 1

communities = 0
total_communities = 0
for k,v in clusters.items():
    for i in v[0]:
        total_communities += 1
        communities += len(i)

summary = open("summary.txt","w")

summary.write("Number of users collected: %d \n Number of messages collected: %d \n"
              "Number of communities discovered: %d \n Average users per community: %d \n"
              "Instances of male class: %d \n Instances of female class: %d \n"
              "Instances of positive sentiment: %d \n Instances of negative sentiment: %d\n"
              "Instances of Neutral sentiment: %d \n Example of male name: %s\n"
              "Example of female name: %s \n "
              "Example of tweet with positive sentiment: %s \nExample of tweet with negative sentiment:%s\n"
              "Example of tweet with neutral sentiment: %s \n " % (len(users_count),len(msg_count), total_communities
                                                                   ,int(communities/total_communities),male,female,pos,
                                                                   neg,neu,classes[2]['male_name'].split()[0],classes[2]['female_name'].split()[0],
                                                                   classes[2]['positive'],classes[2]['negative'],
                                                                   classes[2]['Neutral']))

print("data written to summary.txt\n\n")
print("Number of users collected: ",len(users_count))
print("Number of messages collected: ",len(msg_count))
print("Number of communities discovered: ", total_communities)
print("Average users per community: ",int(communities/total_communities))
print("Instances of male class: ", male)
print("Instances of female class: ", female)
print("Instances of positive sentiment: ", pos)
print("Instances of negative sentiment: ", neg)
print("Instances of Neutral sentiment: ", neu)
print("Example of male name: ",classes[2]['male_name'].split()[0])
print("Example of female name: ",classes[2]['female_name'].split()[0])
print("Example of tweet with positive sentiment:",classes[2]['positive'])
print("Example of tweet with negative sentiment:",classes[2]['negative'])
print("Example of tweet with neutral sentiment:",classes[2]['Neutral'])

