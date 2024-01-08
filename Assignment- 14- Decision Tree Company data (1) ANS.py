#!/usr/bin/env python
# coding: utf-8

# In[1]:


#  Assignments


# In[2]:


# Imoprt Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import  DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report


# In[3]:


# Load data sets
company_data = pd.read_csv("\\Users\\Rohit\\Downloads\\Company_Data.csv")
company_data.head(10)


# In[4]:


company_data.info()


# In[5]:


company_data.corr()


# In[6]:


sns.jointplot(company_data['Sales'],company_data['Income'])


# In[7]:


company_data.loc[company_data["Sales"] <= 10.00,"Sales1"]="Not High"
company_data.loc[company_data["Sales"] >= 10.01,"Sales1"]="High"


# In[8]:


company_data


# # Label Encoding

# In[9]:


from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()


# In[10]:


company_data["ShelveLoc"] = label_encoder.fit_transform(company_data["ShelveLoc"])
company_data["Urban"] = label_encoder.fit_transform(company_data["Urban"])
company_data["US"] = label_encoder.fit_transform(company_data["US"])
company_data["Sales1"] = label_encoder.fit_transform(company_data["Sales1"])


# In[11]:


company_data


# In[12]:


# Define x & y
x = company_data.iloc[:,1:11]
y = company_data['Sales1']


# In[13]:


# Splitting data into training and testing data set
x_train, x_test,y_train,y_test = train_test_split(x,y, test_size=0.2,random_state=50)


# # Building Decision Tree Classifier using Entropy Criteria

# In[14]:


model = DecisionTreeClassifier(criterion = 'entropy',max_depth=3)
model.fit(x_train,y_train)


# In[15]:


model.get_n_leaves()


# In[16]:


preds = model.predict(x_test) # predicting on test data set 
pd.Series(preds).value_counts() # getting the count of each category 


# In[17]:


preds


# In[18]:


pd.crosstab(y_test,preds) # getting the 2 way table to understand the correct and wrong predictions


# In[19]:


# Accuracy 
np.mean(preds==y_test)


# In[20]:


print(classification_report(preds,y_test))


# # Building Decision Tree Classifier (CART) using Gini Criteria

# In[21]:


model_gini = DecisionTreeClassifier(criterion='gini', max_depth=3)


# In[22]:


model_gini.fit(x_train, y_train)


# In[23]:


model_gini.get_n_leaves()


# In[24]:


preds = model_gini.predict(x_test) # predicting on test data set 
pd.Series(preds).value_counts() # getting the count of each category 


# In[25]:


preds


# In[26]:


# Accuracy 
np.mean(preds==y_test)


# In[27]:


print(classification_report(preds,y_test))


# # Building Decision Tree Regression 

# In[28]:


from sklearn.tree import DecisionTreeRegressor


# In[29]:


model_R = DecisionTreeRegressor()
model_R.fit(x_train, y_train)


# In[30]:


preds = model_R.predict(x_test) 


# In[31]:


np.mean(preds==y_test)


# # Plot Tree Diagram

# In[32]:


# Decision Tree Classifier using Entropy Criteria
fig = plt.figure(figsize=(25,20))
fig = tree.plot_tree(model)


# In[33]:


# Decision Tree Classifier (CART) using Gini Criteria
fig = plt.figure(figsize=(25,20))
fig = tree.plot_tree(model_gini)


# In[34]:


# Decision Tree Regression
fig = plt.figure(figsize=(25,20))
fig = tree.plot_tree(model_R)

