# -*- coding: utf-8 -*
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import re
# importing the datasets
data_train = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv' )

print(data_train.iloc[1 , 3])

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

def text_morpher(data):
    corpus = []
    for i in range(0 , int(data.shape[0])):
        tweet = re.sub('[^a-zA-Z]' , " " ,data['text'][i])
        tweet = tweet.lower()
        tweet = tweet.split()
        ps = PorterStemmer()
        tweet = [ps.stem(word) for word in tweet if not word in set(stopwords.words('english'))]
        tweet = ' '.join(tweet)
        corpus.append(tweet)
    return(corpus)
    
# cleaning the text columns----
corpus_tr = text_morpher(data_train)

corpus_te = []
for i in range(0, 3263):
    tweet = re.sub('[^a-zA-Z]' , " " ,data_test['text'][i])
    tweet = tweet.lower()
    tweet = tweet.split()
    ps1 = PorterStemmer()
    tweet = [ps1.stem(word) for word in tweet if not word in set(stopwords.words('english'))]
    tweet = ' '.join(tweet)
    corpus_te.append(tweet)


"""# applying the bag of words model 
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 18000)
x = cv.fit_transform(corpus_tr).toarray()
y = cv.transform(corpus_te).toarray"""

# applying tf-idf 
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features = 16000 , stop_words = 'english') 
x = tfidf.fit_transform(corpus_tr).toarray()
y = tfidf.transform(corpus_te).toarray()
Y = data_train.iloc[: , -1]

# applying the classification model
from sklearn.model_selection import train_test_split
x_train , x_valid , y_train , y_valid = train_test_split(x , Y , test_size = 0.2 , random_state = 42 )


# XGBClassifier
from xgboost import XGBClassifier
classifier = XGBClassifier(n_jobs = -1)
classifier.fit(x_train , y_train)


y_pred = classifier.predict(x_valid)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_valid , y_pred)

from sklearn import metrics
print("accuracy_score" , metrics.accuracy_score(y_valid , y_pred))

y_final = classifier.predict(y)

output=pd.DataFrame(data={'id': data_test['id'] , 'target' : y_final})
  
output.to_csv(path_or_buf = "prediction_nlp.csv" , index=False , sep=',')
 




