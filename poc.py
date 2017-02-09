# -*- coding: utf-8 -*-



import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import svm
import numpy as np
from sklearn import model_selection
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from nltk.stem.porter import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB




df1 = pd.read_csv('D:\\Vodafone\\POC\\CrawlerOutput_1.csv')
df2 = pd.read_csv('D:\\Vodafone\\POC\\CrawlerOutput_2.csv')
df3 = pd.read_csv('D:\\Vodafone\\POC\\CrawlerOutput_3.csv')
df4 = pd.read_csv('D:\\Vodafone\\POC\\CrawlerOutput_4.csv')
df5 = pd.read_csv('D:\\Vodafone\\POC\\CrawlerOutput_5.csv')

dataset = df1.append(df2)
dataset = dataset.append(df3)
dataset = dataset.append(df4)
dataset = dataset.append(df5)

dataset.fillna('')

temp = dataset['ArticleStory'].tolist()

#df = pd.DataFrame(temp)

#tokenized_words = [word_tokenize(i) for i in temp]

#temp.head(2)

#dataset = dataset.reset_index()

#tokenized_docs_no_stopwords = []
#for word in temp:
#    if not word in stopwords.words('english'):
#        tokenized_docs_no_stopwords.append(word)
#    
#            
#print tokenized_docs_no_stopwords


#x= df.apply(re.sub("\d","",df),1)

temp2 = list(map(lambda x : re.sub("\d","",str(x)),temp))


out = []
exception = []
for i in temp:
    try:
        a=  re.sub("\d",'',str(i))
        out.append(a)
    except:
        exception.append(i)
    
#dataset = pd.DataFrame(temp2)

#Stemming
ps = PorterStemmer()
ps.stem('cars')


empty = []
for i in range(len(temp2)):
    a=[]
    for word in temp2[i].split():
        a.append(ps.stem(word))
    empty.append(a)
    

Stemmed = [' '.join(x) for x in empty]

#Lemitization
from nltk.stem.wordnet import WordNetLemmatizer
wordnet = WordNetLemmatizer()

emptylist = []
for i in range(len(temp2)):
    a=[]
    for word in temp2[i].split():
        a.append(wordnet.lemmatize(word))
    emptylist.append(a)
    

lemit_story = [' '.join(x) for x in emptylist]

#DF matrix without Stemming and Lemiitzation
v = TfidfVectorizer(stop_words = 'english')
x = v.fit_transform(temp2)
tf_idf=x.toarray()
xterms = v.get_feature_names()

#-------------------- Kmeans---------------------- 

km = KMeans(n_clusters=5)
km.fit(tf_idf)
clusters = km.labels_.tolist()


X_train, X_test, y_train, y_test = train_test_split(
    tf_idf, dataset['Label'], test_size=0.33, random_state=42)

#---------------------SVM(linear)------------------------
clf = svm.SVC(kernel='linear')
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
#accuracy_score(y_test,y_pred)
clf.score(X_test,y_test)
#0.88

#---------------------SVM(poly)------------------------
clf = svm.SVC(kernel='poly')
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
#accuracy_score(y_test,y_pred)
clf.score(X_test,y_test)
#0.29

#---------------------SVM(sigmoid)------------------------
clf = svm.SVC(kernel='sigmoid')
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
#accuracy_score(y_test,y_pred)
clf.score(X_test,y_test)
#0.29

#--------------------------------RandomForest--------

rf = RandomForestClassifier(n_estimators=100) # initialize
rf.fit(X_train, y_train) # fit the data to the algorithm
rf.score(X_test,y_test)
#0.858

#--------------------------------Naive Bayes--------
mnb = MultinomialNB()
mnb.fit(X_train, y_train) # fit the data to the algorithm
mnb.score(X_test,y_test)
#0.66




    
    
