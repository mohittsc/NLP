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
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression





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
print "Naive Bayes Score = ",mnb.score(X_test,y_test)

#--------------------------------Logistic Regression--------
modelLogistic=LogisticRegression()
Logistic=modelLogistic.fit(X_train, y_train)
print "Logistic Regression Score = ",Logistic.score(X_test,y_test)
#0.80

# Grid Search

import numpy as np
from scipy.stats import uniform as sp_rand
from sklearn.model_selection import RandomizedSearchCV

# prepare a uniform distribution to sample for the alpha parameter
param_grid = {'alpha': sp_rand()}

# create and fit a ridge regression model, testing random alpha values
model = RandomForestClassifier()
rsearch = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=100)
rsearch.fit(X_train, y_train)
print(rsearch)

#Logistic Regression Grid Search

param_grid={'C':(0.01,0.1,1,10,100,1000),
                   'solver':('newton-cg', 'lbfgs'),
                   'multi_class':('multinomial','ovr')}
grid_Logistic=GridSearchCV(modelLogistic,param_grid,n_jobs=-1)
grid_Logistic=grid_Logistic.fit(X_train,y_train)

#Score test set
print "Logistic Regression Score = ",grid_Logistic.score(X_test,y_test)
print "Best Parameters = ",grid_Logistic.best_params_



#cross validation

from sklearn.model_selection import cross_val_score
clf = svm.SVC(kernel='linear')
scores = cross_val_score(clf,X_train, y_train,cv=5)    
scores

logisticmodel=LogisticRegression(multi_class='ovr',C=10,solver='newton-cg')
print "Logistic Regression cv score = ", np.mean(cross_val_score(logisticmodel,X_train, y_train,cv=5))


