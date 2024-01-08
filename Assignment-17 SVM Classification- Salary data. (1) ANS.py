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


# In[4]:


train = pd.read_csv("\\Users\\Rohit\\Downloads\\SalaryData_Train(1).csv")
train.head(10)


# In[5]:


train.info()


# In[6]:


test = pd.read_csv("\\Users\\Rohit\\Downloads\\SalaryData_Test(1).csv")
test.head(10)


# In[7]:


test.info()


# # Preprocessing and Encoding

# In[8]:


from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()


# In[9]:


train["workclass"] = label_encoder.fit_transform(train["workclass"])
train["education"] = label_encoder.fit_transform(train["education"])
train["maritalstatus"] = label_encoder.fit_transform(train["maritalstatus"])
train["occupation"] = label_encoder.fit_transform(train["occupation"])
train["relationship"] = label_encoder.fit_transform(train["relationship"])
train["race"] = label_encoder.fit_transform(train["race"])
train["sex"] = label_encoder.fit_transform(train["sex"])
train["native"] = label_encoder.fit_transform(train["native"])
train["Salary"] = label_encoder.fit_transform(train["Salary"])
train.head()


# In[10]:


test["workclass"] = label_encoder.fit_transform(test["workclass"])
test["education"] = label_encoder.fit_transform(test["education"])
test["maritalstatus"] = label_encoder.fit_transform(test["maritalstatus"])
test["occupation"] = label_encoder.fit_transform(test["occupation"])
test["relationship"] = label_encoder.fit_transform(test["relationship"])
test["race"] = label_encoder.fit_transform(test["race"])
test["sex"] = label_encoder.fit_transform(test["sex"])
test["native"] = label_encoder.fit_transform(test["native"])
test["Salary"] = label_encoder.fit_transform(test["Salary"])
test.head()


# In[11]:


# Define X & y train test


# In[12]:


X_train = train.iloc[:,:-1]
X_train.head()


# In[13]:


y_train = train.iloc[:,-1]
y_train.head()


# In[14]:


X_test = train.iloc[:,:-1]
X_test.head()


# In[15]:


y_test = train.iloc[:,-1]
y_test.head()


# # Grid Search CV

# In[ ]:


clf = SVC()
param_grid = [{'kernel':['rbf'],'gamma':[50,5,10],'C':[15,14,13,12] }]
gsv = GridSearchCV(clf,param_grid,cv=10)
gsv.fit(X_train,y_train)


# # SVM Classification

# In[ ]:


clf = SVC(C= 15, gamma = 5)
clf.fit(X_train , y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred) * 100
print("Accuracy =", acc)
confusion_matrix(y_test, y_pred)


# In[ ]:


salary = pd.merge(train,test)


# In[ ]:


salary


# In[ ]:


X=salary.iloc[:,:-1]
X


# In[ ]:


y=salary.iloc[:,-1]
y


# In[ ]:


clf1 = SVC(C= 15, gamma = 50)
clf1.fit(X , y)
y_pred = clf1.predict(X)
acc1 = accuracy_score(y, y_pred) * 100
print("Accuracy =", acc1)
confusion_matrix(y, y_pred)


# In[ ]:





# In[ ]:




