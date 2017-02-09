
# coding: utf-8

# In[3]:

import pandas as pd


# In[4]:

df1 = pd.read_csv("D:\NLP Python\CrawlerOutput_1.csv")
df2 = pd.read_csv("D:\NLP Python\CrawlerOutput_2.csv")
df3 = pd.read_csv("D:\NLP Python\CrawlerOutput_3.csv")
df4 = pd.read_csv("D:\NLP Python\CrawlerOutput_4.csv")
df5 = pd.read_csv("D:\NLP Python\CrawlerOutput_5.csv")


# Importing and Concatenating

# In[15]:

df=df1.append(df2,df3,df4,df5, ignore_index=true); df


# In[94]:

frames=[df1,df2,df3,df4,df5]
df=pd.concat(frames)


# # Cleaning the text

# In[26]:

import re


# In[32]:

test="mayank1234&*"
print test


# In[35]:

test1 = re.sub("\d+", "", test)
print test1


# In[36]:

test2 = re.sub("[^A-Za-z0-9]+", "", test1)
print test2


# In[74]:

print t1.(lambda x:re.sub("\d+","",x),axis=1)


# In[70]:

print re.sub("[^0-9]","",test)


# In[228]:

df


# In[103]:

df=df.fillna("")


# In[123]:

df["C_Story"]=df.apply(lambda x: re.sub("[\d+]","",x["ArticleStory"]),axis=1)


# In[156]:

df["C_Story"]=df.apply(lambda x: re.sub("[^A-Za-z0-9-\s]+","",x["C_Story"].lower()),axis=1)


# In[155]:

df["C_Title"]=df.apply(lambda x: re.sub("[^A-Za-z0-9-\s]+","",x["ArticleTitle"]),axis=1)
df["C_Title"]=df.apply(lambda x: re.sub("[^A-Za-z0-9-\s]+","",x["C_Title"].lower()),axis=1)


# # lemmatization

# In[222]:

wordforlem=df.apply(lambda x:x["C_Story"].split(),axis=1)


# In[ ]:

from nltk.stem.wordnet import WordNetLemmatizer as lmtzr


# In[217]:

L_word=wordforlem.map(lambda x:[lm.lemmatize(y) for y in x])


# In[220]:

L_word_join=L_word.map(lambda x:" ".join(x))


# In[328]:

L_word_join


# In[164]:

from sklearn.feature_extraction.text import TfidfVectorizer


# In[223]:

tfidf=TfidfVectorizer(stop_words='english',ngram_range=(1,1))


# In[224]:

sparsematrix=tfidf.fit_transform(L_word_join)


# training and testing data set

# In[245]:

from sklearn.cross_validation import train_test_split


# In[246]:

X_train, X_test, y_train, y_test = train_test_split(df["Label"], sparsematrix, test_size=0.2, random_state=0)


# # NB

# In[ ]:

from sklearn.naive_bayes import MultinomialNB


# In[244]:

clf = MultinomialNB().fit(y_train, X_train)


# In[248]:

#for prediction
clf.score(y_test,X_test)


# # Grid Search

# In[268]:

param_grid={'alpha':(0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0)}


# In[265]:

#from sklearn.model_selection import ParameterGrid
from sklearn.grid_search  import GridSearchCV


# In[275]:

a = GridSearchCV(clf,param_grid,n_jobs=1)


# In[276]:

it_m=a.fit(y_train, X_train)


# In[278]:

it_m.best_estimator_


# # k-cross validation

# In[297]:

from sklearn.cross_validation import KFold
from sklearn import cross_validation


# In[324]:

kc=KFold(y_train.shape[0],n_folds=10,shuffle=True,random_state=0)


# In[329]:

cv1= cross_validation.cross_val_score(clf, sparsematrix, df["Label"], cv=kc)


# In[332]:

cv1


# In[333]:




# In[279]:

class A:
    def __init__(self):
        pass
    def do_something(self,x):
        print x
        


# In[ ]:




# In[188]:

from nltk.stem.wordnet import WordNetLemmatizer
lmtzr = WordNetLemmatizer()


# In[200]:

lm = lmtzr()
lm.lemmatize("cars")


# In[192]:

test=lmtzr.lemmatize("durable")


# In[186]:

lmtdata=sparsematrix.apply(lambda x:lmtzr(x["sparsematrix"]),axis=1)


# In[175]:




# In[158]:

#check
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(df.C_Story)


# In[160]:

X_train_counts.data


# In[152]:

import nltk

from nltk import stopwords
df["C_Story"]=df.apply(lambda x:re.sub("stopwords"))

#stopword = stopwords.stopword()

#clean_words = [x for x in datalist if x not in stopword]


# In[132]:

wordlist=df.apply(lambda x:re.split("\s+",x["C_Story"]),axis=1)


# In[135]:

wordlist.lower()


# In[148]:

C_wordlist=",".join(df["C_Story"])


# In[145]:

C_wordlist=re.sub("\s+-\n",",",C_wordlist)


# In[150]:

df["C_Story"]


# In[149]:

type(C_wordlist)


# In[ ]:



