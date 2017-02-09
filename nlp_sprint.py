
# coding: utf-8

# In[1]:

import pandas as pd



# In[2]:

get_ipython().magic(u'pinfo pd.read_csv')


# In[7]:

crawl1 = pd.read_csv("C:\Users\debayan.daw\Documents\Python folder\NLP\CrawlerOutput_1")


# In[8]:

crawl1 = pd.read_csv("C:\Users\debayan.daw\Documents\Python folder\NLP\CrawlerOutput_1.csv")


# In[9]:

crawl2 = pd.read_csv("C:\Users\debayan.daw\Documents\Python folder\NLP\CrawlerOutput_2.csv")


# In[10]:

crawl3 = pd.read_csv("C:\Users\debayan.daw\Documents\Python folder\NLP\CrawlerOutput_3.csv")


# In[11]:

crawl4 = pd.read_csv("C:\Users\debayan.daw\Documents\Python folder\NLP\CrawlerOutput_4.csv")


# In[12]:

crawl5 = pd.read_csv("C:\Users\debayan.daw\Documents\Python folder\NLP\CrawlerOutput_5.csv")


# In[10]:

pd.describe(crawl1)


# In[13]:

frames = [crawl1, crawl2, crawl3,crawl4,crawl5]


# In[14]:

nlp = pd.concat(frames)


# In[13]:

nlp.describe


# In[14]:

nlp[0:3] 


# In[16]:

a = 1


# In[18]:

b=1


# In[19]:

id(a)


# In[20]:

id(b)


# In[21]:

nlp.dtypes


# In[22]:

len(nlp)


# In[15]:

import re


# In[24]:

abc = "abc number.  jap usd , ani$$ "


# In[26]:

abc


# In[27]:

clear = abc.filter(regex = 'regex')


# In[28]:

clear = pd.filter(regex = 'abc')


# In[29]:

print re.sub("."," " , abc)


# In[54]:

replace = re.sub('[\.,$"  " ]'," " , abc)


# In[55]:

replace


# In[44]:

abc


# In[45]:

replace = re.sub(".","" , abc)


# In[46]:

replace


# In[35]:

replace = re.sub('.',"" , abc)


# In[40]:

s= "asdfasdfv "
print s
print s[:5]


# In[37]:

column1 = nlp['ArticleTitle']


# In[38]:

col1_change = nlp.apply(lambda x:x['ArticleTitle'],axis = 1)


# In[61]:

lst = ['abc$def','str123','place**','match']
b = re.sub("$","",for i in L)


# In[62]:

df = pd.DataFrame(columns= ['A','B'],data=[[2,4],[3,6],[4,8]])


# In[16]:

nlp


# In[64]:

df


# In[66]:

df.assign(SumAB= df['A']+df['B'])


# In[69]:

def sum1(x):
    return x['A']+x['B']
df['Value'] = df.apply(lambda row: sum1, axis=1)


# In[70]:

df


# In[71]:

df['Value'] = df.apply(lambda row: sum1(df), axis=1)


# In[72]:

df


# In[74]:

df['Value'] = df.apply(lambda row: sum1(df), axis=1)


# In[78]:

df['Value'] = df.apply(sum1, axis=1)


# In[79]:

df


# In[17]:

def str_replace(x):
    return (re.sub('\W+'," ",x['ArticleTitle']))

nlp['Article_Title_clean'] = nlp.apply(str_replace, axis=1)


# In[18]:

nlp


# In[19]:

def str_replace1(x):
    return (re.sub('\W+'," ",str(x['ArticleStory'])))
nlp['Article_Story_clean'] = nlp.apply(str_replace1, axis=1)


# In[20]:

nlp


# In[101]:

def str_replace1(x):
    return (re.sub('\W+'," ",x['ArticleStory']))
nlp['Article_Story_clean'] = nlp.apply(str_replace1, axis=1)


# In[102]:

del nlp['Value']


# In[103]:

nlp


# In[106]:


def str_replace1(x):
    return (re.sub('\W+'," ",x['ArticleStory']))
nlp['Article_Story_clean'] = nlp.apply(str_replace1, axis=1)


# In[21]:

nlp.ix[185]
nlp.reset_index()


# In[113]:

nlp = nlp.reset_index()


# In[115]:

nlp.ix[185]


# In[22]:

nlp


# In[23]:

from sklearn import feature_extraction


# In[27]:

tfidf = feature_extraction.text.TfidfVectorizer(stop_words='english',ngram_range=(1,2),
                                               min_df=.005)


# In[28]:

art1 = tfidf.fit_transform(nlp['Article_Title_clean'])


# In[29]:

art1


# In[123]:

Art_1 = sklearn.feature_extraction.text.TfidfVectorizer(input=nlp['Article_Title_clean'], encoding=u'utf-8', 
                                                decode_error=u'strict', 
                                                strip_accents=None, lowercase=True, preprocessor=None,
                                                tokenizer=None, analyzer=u'word', stop_words='English', 
                                                token_pattern=u'(?u)\b\w\w+\b',
                                                ngram_range=(1, 1), max_df=1.0, min_df=1, max_features=None, 
                                                vocabulary=None, binary=False, 
                                                dtype=<type 'numpy.int64'>, norm=u'l2', 
                                                use_idf=True, smooth_idf=True, sublinear_tf=False)


# In[ ]:

art1 = sklearn.feature


# In[124]:

a = sklearn.feature_extraction.text.TfidfVectorizer(nlp['Article_Title_clean'])


# In[30]:

tfidf = feature_extraction.text.TfidfVectorizer(stop_words='english',ngram_range=(1,2),
                                               min_df=.005)


# In[33]:

art2 = tfidf.fit_transform(nlp['Article_Story_clean'])


# In[35]:

art2


# In[36]:

nlp


# In[39]:

nlp_class = tfidf.fit_transform(nlp['Label'])


# In[40]:

nlp_class


# In[48]:

tfid1 = feature_extraction.text.TfidfVectorizer(stop_words='english')


# In[50]:

nlp_class = tfid1.fit_transform(nlp['Label'])


# In[51]:




# In[58]:

import pandas as pd
art1_df = pd.DataFrame(art1)


# In[59]:

class A:
    def __init__(self):
        pass
    def do_something(self,x):
        return (x+1)


# In[60]:

a = A()


# In[61]:

print a.do_something(2)


# In[74]:

art1_data = art1.toarray()


# In[76]:

art2_data =  art2.toarray()
art2_data


# In[77]:

from sklearn.model_selection import train_test_split


# In[80]:

X1_train, X1_test, y1_train, y1_test = train_test_split(art1_data,nlp['Label'], test_size=0.30, random_state=100)


# X2_train, X2_test, y2_train, y2_test = train_test_split(art2_data,nlp['Label'], test_size=0.30, random_state=100)

# In[81]:

X2_train, X2_test, y2_train, y2_test = train_test_split(art2_data,nlp['Label'], test_size=0.30, random_state=100)


# In[89]:

#svm
from sklearn import svm


# In[197]:

model = svm.SVC(kernel='rbf', C=1, gamma=2)


# In[198]:

model.fit(X1_train, y1_train)
model.score(X1_train, y1_train)
#predicted= model.predict(X2_test)


# In[201]:

predicted1= model.predict(X1_test)


# In[202]:

model.score(X1_test,y1_test)


# In[203]:

model.fit(X2_train, y2_train)
model.score(X2_train, y2_train)


# In[204]:

predicted2= model.predict(X2_test)


# In[205]:

model.score(X2_test,y2_test)


# In[238]:

y_true, y_pred = y2_test, model.predict(X2_test)
print(classification_report(y_true, y_pred))


# In[143]:

#Naive bayes
from sklearn.naive_bayes import MultinomialNB


# In[144]:

model = MultinomialNB()


# In[ ]:

model.fit(X1_train, y1_train)
model.score(X1_train, y1_train)


# In[147]:

predicted1= model.predict(X1_test)


# In[148]:

model.score(X1_test,y1_test)


# In[149]:

model.fit(X2_train, y2_train)
model.score(X2_train, y2_train)


# In[150]:

predicted2= model.predict(X2_test)


# In[151]:

model.score(X2_test,y2_test)


# In[226]:

from sklearn.naive_bayes import GaussianNB


# In[227]:

model = GaussianNB()


# In[228]:

model.fit(X1_train, y1_train)
model.score(X1_train, y1_train)


# In[229]:

predicted1= model.predict(X1_test)


# In[230]:

model.score(X1_test,y1_test)


# In[231]:

model.fit(X2_train, y2_train)
model.score(X2_train, y2_train)


# In[232]:

predicted2= model.predict(X2_test)


# In[233]:

model.score(X2_test,y2_test)


# In[239]:

y_true, y_pred = y2_test, model.predict(X2_test)
print(classification_report(y_true, y_pred))


# In[194]:

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


# In[188]:

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0,1, 2],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'],'gamma': [0,1, 2], 'C': [1, 10, 100, 1000]},
                    {'kernel': ['poly'],'gamma': [0,1, 2], 'C': [1, 10, 100, 1000]}]


# In[163]:

scores = ['precision', 'recall']


# In[189]:

model  = GridSearchCV(SVC(), tuned_parameters, cv=5)


# In[190]:

#Article Title
model.fit(X1_train, y1_train)


# In[183]:

model.best_params_


# In[184]:

means = model.cv_results_['mean_test_score']
stds = model.cv_results_['std_test_score']


# In[185]:

means


# In[186]:

stds


# In[191]:

model.cv_results_


# In[195]:

y_true, y_pred = y_test, model.predict(X1_test)
print(classification_report(y_true, y_pred))


# In[196]:

#Article Story
model.fit(X2_train, y2_train)


# In[ ]:

model.best_params_


# In[ ]:

means = model.cv_results_['mean_test_score']
stds = model.cv_results_['std_test_score']


# In[ ]:

model.cv_results_


# In[ ]:

y_true, y_pred = y_test, model.predict(X1_test)
print(classification_report(y_true, y_pred))


# In[206]:

#K fold Cross validation
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# In[207]:

#Article story SVC
n_folds = 5
this_scores = cross_val_score(SVC(kernel='rbf', C=1, gamma=2), X2_train, y2_train, cv=n_folds, n_jobs=1)


# In[221]:

Accuracy, Deviation = this_scores.mean(), this_scores.std() * 2


# In[222]:

Accuracy, Deviation


# In[223]:

#Article title
n_folds = 5
this_scores = cross_val_score(SVC(kernel='rbf', C=1, gamma=2), X1_train, y1_train, cv=n_folds, n_jobs=1)


# In[224]:

Accuracy, Deviation = this_scores.mean(), this_scores.std() * 2


# In[225]:

Accuracy, Deviation


# In[234]:

#Article story Naive bayes
n_folds = 5
this_scores = cross_val_score(GaussianNB(), X2_train, y2_train, cv=n_folds, n_jobs=1)


# In[235]:

Accuracy, Deviation = this_scores.mean(), this_scores.std() * 2


# In[236]:

Accuracy, Deviation

