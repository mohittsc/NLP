
# coding: utf-8

# ### Loading libraries

# In[88]:

import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
#nltk.download()


# ### Importing data and consolidating

# In[45]:

fileList=[]
for fileIndex in xrange(1,6):
    path='D:/Python Training/POC/Raw Data/CrawlerOutput_%i.csv'% (fileIndex)
    fileList.append(pd.read_csv(path))
    
rawData=pd.concat(fileList)


# In[46]:

features=rawData.loc[:,['ArticleStory']]
labels=rawData.Label


# In[47]:

labEncoder=LabelEncoder()
labelsEncoded=labEncoder.fit_transform(labels)


# ### Cleaning, Lemmatization, Tokenization, Stopwords, Tf-idf

# In[48]:

features=features.replace(np.nan,'')
features['Cleaned']=features.ArticleStory.map(lambda row: re.sub('\s+',' ',re.sub('[^a-z]', ' ',row.lower())).strip())

lemmatizer=WordNetLemmatizer()
features['Lemmatized']=features.Cleaned.map(lambda row: ' '.join([lemmatizer.lemmatize(word) for word in row.split()]))


# In[49]:

#Breaking into train and test set
X_train, X_test, Y_train, Y_test = train_test_split(features.Lemmatized,labelsEncoded,test_size=0.2,random_state=0)


# In[50]:

tfIdfVect=TfidfVectorizer(stop_words="english",ngram_range=(1,2),max_df=0.95,min_df=0.05)
X_train=tfIdfVect.fit_transform(X_train)


# In[51]:

X_test_transformed=tfIdfVect.transform(X_test)


# ### Training the Model

# #### Naive Bayes 

# In[52]:

modelNB=MultinomialNB()
modelNB=modelNB.fit(X_train,Y_train)

#Scoring using test set
print "Naive Bayes Score = ",modelNB.score(X_test_transformed,Y_test)


# #### SVM with linear kernel 

# In[53]:

modelLinearSVM=SGDClassifier()
modelLinearSVM=modelLinearSVM.fit(X_train,Y_train)

#Scoring using test set
print "Linear SVM Score = ",modelLinearSVM.score(X_test_transformed,Y_test)


# #### Random Forest 

# In[54]:

modelRandomForest=RandomForestClassifier(n_estimators=100)
modelRandomForest=modelRandomForest.fit(X_train,Y_train)

#Scoring using test set
print "Random Forest Score = ",modelRandomForest.score(X_test_transformed,Y_test)


# #### Logistic Regression

# In[55]:

modelLogistic=LogisticRegression()
modelLogistic=modelLogistic.fit(X_train,Y_train)

#Scoring using test set
print "Logistic Regression Score = ",modelLogistic.score(X_test_transformed,Y_test)


# ### Parameter Optimization (Grid Search)

# #### Naive Bayes 

# In[56]:

parametersNB={'alpha':(0,0.2,0.4,0.6,0.8,1)}
gs_NB=GridSearchCV(modelNB,parametersNB,n_jobs=-1)
gs_NB=gs_NB.fit(X_train,Y_train)

#Scoring using test set
print "Naive Bayes Score = ",gs_NB.score(X_test_transformed,Y_test)
print "Best Parameters = ",gs_NB.best_params_


# #### SVM with linear kernel 

# In[57]:

parametersSVM={'loss':('hinge','log','modified_huber','perceptron'),
              'penalty':('l1','l2'),
              'alpha':(0.00005,0.0001,0.0005,0.001,0.005,0.01),
              'n_iter':(1,5,10,50,100)}
gs_SVM=GridSearchCV(modelLinearSVM,parametersSVM,n_jobs=-1)
gs_SVM=gs_SVM.fit(X_train,Y_train)

#Scoring using test set
print "Linear SVM Score = ",gs_SVM.score(X_test_transformed,Y_test)
print "Best Parameters = ",gs_SVM.best_params_


# #### Random Forest 

# In[58]:

parametersRF={'n_estimators':(5,10,25,50,100,500,1000)}
gs_RF=GridSearchCV(modelRandomForest,parametersRF,n_jobs=-1)
gs_RF=gs_RF.fit(X_train,Y_train)

#Scoring using test set
print "Random Forest Score = ",gs_RF.score(X_test_transformed,Y_test)
print "Best Parameters = ",gs_RF.best_params_


# #### Logistic Regression

# In[59]:

parametersLogistic={'C':(0.01,0.1,1,10,100,1000),
                   'solver':('newton-cg', 'lbfgs'),
                   'multi_class':('multinomial','ovr')}
gs_Logistic=GridSearchCV(modelLogistic,parametersLogistic,n_jobs=-1)
gs_Logistic=gs_Logistic.fit(X_train,Y_train)

#Scoring using test set
print "Logistic Regression Score = ",gs_Logistic.score(X_test_transformed,Y_test)
print "Best Parameters = ",gs_Logistic.best_params_


# ### Model Selection (K-Fold Cross Validation)

# In[72]:

#Consolidated features and label dataset
X=tfIdfVect.fit_transform(features.Lemmatized)
Y=labelsEncoded

kfIter=KFold(n=X.shape[0],n_folds=5,shuffle=True,random_state=0)


# In[76]:

#Obtaining cross validation score for each model
finalModelNB=MultinomialNB(alpha=0.2)
print "Naive Bayes Cross-Validation = ", np.mean(cross_val_score(finalModelNB,X,Y,cv=kfIter))

finalModelSVM=SGDClassifier(penalty='l2',alpha=0.0005,n_iter=10,loss='modified_huber')
print "Linear SVM Cross-Validation = ", np.mean(cross_val_score(finalModelSVM,X,Y,cv=kfIter))

finalModelRF=RandomForestClassifier(n_estimators=1000)
print "Random Forest Cross-Validation = ", np.mean(cross_val_score(finalModelRF,X,Y,cv=kfIter))

finalModelLogistic=LogisticRegression(multi_class='ovr',C=100,solver='newton-cg')
print "Logistic Regression Cross-Validation = ", np.mean(cross_val_score(finalModelLogistic,X,Y,cv=kfIter))

