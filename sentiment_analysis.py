#Importing various data_test Model

import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import pandas
from sklearn.model_selection import train_test_split
import numpy
tweets= pd.read_csv(r"../input/Tweets.csv") #path to the data
tweets.head()
list(tweets.columns.values)
tweets.head()

#normalizing the data_test
(len(tweets)-tweets.count())/len(tweets)

#Plotting the graph to compare the airline
Mood_count=tweets['airline_sentiment'].value_counts()
Index = [1,2,3]
plt.bar(Index,Mood_count)
plt.xticks(Index,['negative','neutral','positive'],rotation=45)
plt.ylabel('Mood Count')
plt.xlabel('Mood')
plt.title('Count of Moods')
tweets['airline'].value_counts()

#plots
def plot_sub_sentiment(Airline):
    df=tweets[tweets['airline']==Airline]
    count=df['airline_sentiment'].value_counts()
    Index = [1,2,3]
    plt.bar(Index,count)
    plt.xticks(Index,['negative','neutral','positive'])
    plt.ylabel('Mood Count')
    plt.xlabel('Mood')
    plt.title('Count of Moods of '+Airline)
plt.figure(1,figsize=(12, 12))
plt.subplot(231)
plot_sub_sentiment('US Airways')
plt.subplot(232)
plot_sub_sentiment('United')
plt.subplot(233)
plot_sub_sentiment('American')
plt.subplot(234)
plot_sub_sentiment('Southwest')
plt.subplot(235)
plot_sub_sentiment('Delta')
plt.subplot(236)
plot_sub_sentiment('Virgin America')
Airline_count = tweets['airline'].sort_index().value_counts()
Airline_count.plot(kind='bar',rot=45)
plt.show()
air_senti=pd.crosstab(tweets.airline, tweets.airline_sentiment)
air_senti
percent=air_senti.apply(lambda a: a / a.sum() * 100, axis=1)
percent

#changing the time column in the data to readable format
tweets['tweet_created'] = pd.to_datetime(tweets['tweet_created'])
tweets["date_created"] = tweets["tweet_created"].dt.date
tweets["date_created"]

df = tweets.groupby(['date_created','airline'])
df = df.airline_sentiment.value_counts()
df.unstack(0)

#Natural language processor to predict the commonly used words in the comments
import re, nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
wordnet_lemmatizer = WordNetLemmatizer()

def normalizer(tweet):
    only_letters = re.sub("[^a-zA-Z]", " ",tweet)
    tokens = nltk.word_tokenize(only_letters)[2:]
    lower_case = [l.lower() for l in tokens]
    filtered_result = list(filter(lambda l: l not in stop_words, lower_case))
    lemmas = [wordnet_lemmatizer.lemmatize(t) for t in filtered_result]
    return lemmas


normalizer("Here is text about an airline I like.")
pd.set_option('display.max_colwidth', -1) # Setting this so we can see the full content of cells
tweets['normalized_tweet'] = tweets.text.apply(normalizer)
tweets[['text','normalized_tweet']].head()
import collections
def count_words(input):
    cnt = collections.Counter()
    for row in input:
        for word in row:
            cnt[word] += 1
    return cnt

tweets[(tweets.airline_sentiment == 'negative')][['grams']].apply(count_words)['grams'].most_common(20)



import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer
count_vectorizer = CountVectorizer(ngram_range=(1,2))

vectorized_data = count_vectorizer.fit_transform(tweets.text)
indexed_data = hstack((np.array(range(0,vectorized_data.shape[0]))[:,None], vectorized_data))


#Normalizing the data based on the comments
def sentiment2target(sentiment):
    return {
        'negative': 0,
        'neutral': 1,
        'positive' : 2
    }[sentiment]
targets = tweets.airline_sentiment.apply(sentiment2target)
print(f'{targets}')

#datasplitting

from sklearn.model_selection import train_test_split
data_train, data_test, targets_train, targets_test = train_test_split(indexed_data, targets, test_size=0.2, random_state=0)
data_train_index = data_train[:,0]
data_train = data_train[:,1:]
data_test_index = data_test[:,0]
data_test = data_test[:,1:]

#Data classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import metrics
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.ensemble import BaggingClassifier
knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=3)

Classifiers = [
    LogisticRegression(C=0.000000001,solver='liblinear',max_iter=200),
    DecisionTreeClassifier(),
    OneVsRestClassifier(svm.SVC(gamma=0.01, C=100., probability=True, class_weight='balanced', kernel='linear')),
    KNeighborsClassifier(3),
    SVC(kernel="rbf", C=0.025, probability=True),
    AdaBoostClassifier(),
    RandomForestClassifier(n_estimators=250)]
Accuracy=[]
Model=[]

for classifier in Classifiers:

            fit = classifier.fit(data_train, targets_train)
            pred = fit.predict(data_test)
            accuracy = accuracy_score(pred,targets_test)
            Accuracy.append(accuracy)
            Model.append(classifier.__class__.__name__)
            print('Accuracy of '+classifier.__class__.__name__+'is '+str(accuracy))

#Graph to compare various Classifiers
Index = [1,2,3,4,5,6,7]
plt.bar(Index,Accuracy)
plt.xticks(Index, Model,rotation=45)
plt.ylabel('Accuracy')
plt.xlabel('Model')
plt.title('Accuracies of Models')
