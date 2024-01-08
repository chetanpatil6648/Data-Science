#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Assignment


# In[2]:


# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn import preprocessing


# In[3]:


# Load datasets
fraud = pd.read_csv("\\Users\\Rohit\\Downloads\\Fraud_check.csv")
fraud


# In[5]:


fraud.info()


# In[6]:


fraud["TaxInc"] = pd.cut(fraud["Taxable.Income"], bins = [10002,30000,99620], labels = ["Risky", "Good"])
fraud["TaxInc"]


# In[7]:


fraudcheck = fraud.drop(columns=["Taxable.Income"])
fraudcheck 


# In[8]:


FC = pd.get_dummies(fraudcheck .drop(columns = ["TaxInc"]))


# In[9]:


Fraud_final = pd.concat([FC,fraudcheck ["TaxInc"]], axis = 1)


# In[10]:


colnames = list(Fraud_final.columns)
colnames


# In[11]:


predictors = colnames[:9]
predictors


# In[12]:


target = colnames[9]
target


# In[13]:


X = Fraud_final[predictors]
X.shape


# In[14]:


Y = Fraud_final[target]
Y


# In[15]:


# Splitting the data into the Training data and Test data


# In[16]:


from sklearn.model_selection import train_test_split


# In[19]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 40)


# # Feature Scaling

# In[20]:


from sklearn.preprocessing import StandardScaler


# In[21]:


sc = StandardScaler()


# In[22]:


X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# # Training the Random Forest Classification model on the Training data

# In[23]:


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 40)
classifier.fit(X_train, Y_train)


# In[25]:


classifier.fit(X_train, Y_train)


# In[26]:


classifier.score(X_test, Y_test)


# # Predicting the Test set results

# In[28]:


y_pred = classifier.predict(X_test)


# In[29]:


y_pred


# In[30]:


from sklearn.metrics import confusion_matrix, accuracy_score


# In[31]:


cm = confusion_matrix(Y_test, y_pred)


# In[32]:


print(cm)


# In[33]:


accuracy_score(Y_test, y_pred)


# In[ ]:


classifier = RandomForestClassifier(n_estimators=100, criterion='gini')
classifier.fit(_train, y_train)

