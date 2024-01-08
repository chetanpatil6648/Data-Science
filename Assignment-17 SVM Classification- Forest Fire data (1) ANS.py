#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Assignments


# In[2]:


# Import Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score


# In[3]:


# Load data sets
fire = pd.read_csv("\\Users\\Rohit\\Downloads\\forestfires (1).csv")
fire.head()


# In[4]:


fire.info()


# # Preprocessing & Label Encoding

# In[5]:


from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()


# In[6]:


fire["month"] = label_encoder.fit_transform(fire["month"])
fire["day"] = label_encoder.fit_transform(fire["day"])
fire["size_category"] = label_encoder.fit_transform(fire["size_category"])


# In[7]:


fire.head()


# In[8]:


# Define X & y


# In[9]:


X=fire.iloc[:,:11]
X.head()


# In[10]:


y=fire["size_category"]
y.head


# In[11]:


# Split the Data intp Training Data and Test Data


# In[12]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3)


# # Grid Search CV

# In[13]:


clf = SVC()
param_grid = [{'kernel':['rbf'],'gamma':[50,5,10],'C':[15,14,13,12] }]
gsv = GridSearchCV(clf,param_grid,cv=10)
gsv.fit(X_train,y_train)


# In[14]:


gsv.best_params_ , gsv.best_score_


# # SVM Classification

# In[15]:


clf = SVC(C= 15, gamma = 50)
clf.fit(X_train , y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred) * 100
print("Accuracy =", acc)
confusion_matrix(y_test, y_pred)


# In[16]:


clf1 = SVC(C= 15, gamma = 50)
clf1.fit(X , y)
y_pred = clf1.predict(X)
acc1 = accuracy_score(y, y_pred) * 100
print("Accuracy =", acc1)
confusion_matrix(y, y_pred)


# # Grid Search CV Using Poly

# In[17]:


clf2 = SVC()
param_grid = [{'kernel':['poly'],'gamma':[50,5,10,0.5],'C':[15,14,13,12,11,10,0.1,0.001] }]
gsv = GridSearchCV(clf,param_grid,cv=10)
gsv.fit(X_train,y_train)


# In[18]:


gsv.best_params_ , gsv.best_score_


# # Grid Search CV Using Sigmoid

# In[19]:


clf3 = SVC()
param_grid = [{'kernel':['sigmoid'],'gamma':[50,5,10,0.5],'C':[15,14,13,12,11,10,0.1,0.001] }]
gsv = GridSearchCV(clf,param_grid,cv=10)
gsv.fit(X_train,y_train)


# In[20]:


gsv.best_params_ , gsv.best_score_

