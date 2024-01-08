#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Assignments


# In[2]:


# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


# Load data sets


# In[4]:


salary_train = pd.read_csv("\\Users\\Rohit\\Downloads\\SalaryData_Train.csv")
salary_train


# In[5]:


salary_test = pd.read_csv("\\Users\\Rohit\\Downloads\\SalaryData_Test.csv")
salary_test


# In[6]:


salary_train.columns


# In[7]:


salary_test.columns


# In[8]:


salary_test.dtypes


# In[9]:


salary_train.dtypes


# In[10]:


salary_train.info()


# In[11]:


salary_test.info()


# In[12]:


string_columns=['workclass','education','maritalstatus','occupation','relationship','race','sex','native']


# # Graphical Visualization

# In[13]:


sns.pairplot(salary_train)


# In[14]:


sns.pairplot(salary_test)


# In[15]:


sns.boxplot(salary_train['Salary'], salary_train['capitalgain'])


# In[16]:


sns.boxplot(salary_test['Salary'], salary_test['capitalgain'])


# In[17]:


sns.countplot(salary_train['Salary'])


# In[18]:


sns.countplot(salary_test['Salary'])


# In[19]:


plt.figure(figsize=(20,10))
sns.barplot(x='Salary', y='hoursperweek', data=salary_train)
plt.show()


# In[20]:



plt.figure(figsize=(20,10))
sns.barplot(x='Salary', y='hoursperweek', data=salary_test)
plt.show()


# In[21]:


plt.figure(figsize=(15,10))
sns.lmplot(y='capitalgain', x='hoursperweek',data=salary_train)
plt.show()


# In[22]:


plt.figure(figsize=(15,10))
sns.lmplot(y='capitalgain', x='hoursperweek',data=salary_test)
plt.show()


# In[23]:


plt.subplots(1,2, figsize=(16,8))

colors = ["#FF0000", "#64FE2E"]
labels ="capitalgain", "capitalloss"

plt.suptitle('salary of an individual', fontsize=18)

salary_train["Salary"].value_counts().plot.pie(explode=[0,0.25], autopct='%1.2f%%', shadow=True, colors=colors, 
                                             labels=labels, fontsize=10, startangle=25)


# In[24]:


plt.style.use('seaborn-whitegrid')

salary_train.hist(bins=20, figsize=(15,10), color='red')
plt.show()


# In[25]:


plt.subplots(1,2, figsize=(16,8))

colors = ["#FF0000", "#64FE2E"]
labels ="capitalgain", "capitalloss"

plt.suptitle('salary of an individual', fontsize=18)

salary_test["Salary"].value_counts().plot.pie(explode=[0,0.25], autopct='%1.2f%%', shadow=True, colors=colors, 
                                             labels=labels, fontsize=10, startangle=25)


# In[26]:


plt.style.use('seaborn-whitegrid')

salary_test.hist(bins=20, figsize=(15,10), color='red')
plt.show()


# # Preprocessing

# In[27]:


from sklearn import preprocessing
label_encoder=preprocessing.LabelEncoder()


# In[28]:


for i in string_columns:
    salary_train[i]=label_encoder.fit_transform(salary_train[i])
    salary_test[i]=label_encoder.fit_transform(salary_test[i])


# In[29]:


col_names=list(salary_train.columns)
col_names


# In[30]:


train_X=salary_train[col_names[0:13]]
train_X


# In[31]:


train_Y=salary_train[col_names[13]]
train_Y


# In[32]:


test_x=salary_test[col_names[0:13]]
test_x


# In[33]:


test_y=salary_test[col_names[13]]
test_y


# # Build Naive Bayes Model

# # Gaussian Naive Bayes

# In[34]:


from sklearn.naive_bayes import GaussianNB
Gnbmodel=GaussianNB()


# In[35]:


train_pred_gau=Gnbmodel.fit(train_X,train_Y).predict(train_X)
train_pred_gau


# In[36]:


test_pred_gau=Gnbmodel.fit(train_X,train_Y).predict(test_x)
test_pred_gau


# In[37]:


train_acc_gau=np.mean(train_pred_gau==train_Y)


# In[38]:


test_acc_gau=np.mean(test_pred_gau==test_y)


# In[39]:


train_acc_gau


# In[40]:


test_acc_gau


# # Multinomial Naive Bayes

# In[41]:


from sklearn.naive_bayes import MultinomialNB
Mnbmodel=MultinomialNB()


# In[42]:


train_pred_multi=Mnbmodel.fit(train_X,train_Y).predict(train_X)
train_pred_multi


# In[43]:


test_pred_multi=Mnbmodel.fit(train_X,train_Y).predict(test_x)
test_pred_multi


# In[44]:


train_acc_multi=np.mean(train_pred_multi==train_Y)
train_acc_multi


# In[45]:


test_acc_multi=np.mean(test_pred_multi==test_y)
test_acc_multi

