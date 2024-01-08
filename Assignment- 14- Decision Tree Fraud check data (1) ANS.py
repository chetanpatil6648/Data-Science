#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Assignments


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
Fraud_check = pd.read_csv("\\Users\\Rohit\\Downloads\\Fraud_check.csv")
Fraud_check


# In[4]:


Fraud_check.info()


# In[5]:


Fraud_check.corr()


# In[6]:


#Fraud_check.loc[Fraud_check["Taxable.Income"]!="Good","Taxable_Income"]="Risky"
Fraud_check.loc[Fraud_check["Taxable.Income"] <= 30000,"Taxable_Income"]="Good"
Fraud_check.loc[Fraud_check["Taxable.Income"] > 30001,"Taxable_Income"]="Risky"


# In[7]:


Fraud_check


# # Label Encoding

# In[8]:


from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()


# In[9]:


Fraud_check["Undergrad"] = label_encoder.fit_transform(Fraud_check["Undergrad"])
Fraud_check["Marital.Status"] = label_encoder.fit_transform(Fraud_check["Marital.Status"])
Fraud_check["Urban"] = label_encoder.fit_transform(Fraud_check["Urban"])
Fraud_check["Taxable_Income"] = label_encoder.fit_transform(Fraud_check["Taxable_Income"])


# In[10]:


Fraud_check


# In[11]:


Fraud_check.drop(['City.Population'],axis=1,inplace=True)
Fraud_check.drop(['Taxable.Income'],axis=1,inplace=True)


# In[12]:


Fraud_check["Taxable_Income"].unique()


# In[13]:


Fraud_check


# In[14]:


# Define x 
x = Fraud_check.iloc[:,0:4]
x


# In[15]:


# Define y
y = y = Fraud_check["Taxable_Income"]
y


# In[16]:


# Splitting data into training and testing data set
x_train, x_test,y_train,y_test = train_test_split(x,y, test_size=0.2,random_state=40)


# # Building Decision Tree Classifier using Entropy Criteria

# In[17]:


model = DecisionTreeClassifier(criterion = 'entropy')
model.fit(x_train,y_train)


# In[18]:


model.get_n_leaves()


# In[19]:


preds = model.predict(x_test) # predicting on test data set 
pd.Series(preds).value_counts() # getting the count of each category 


# In[20]:


preds


# In[21]:


# Accuracy 
np.mean(preds==y_test)


# In[22]:


print(classification_report(preds,y_test))


# # Building Decision Tree Classifier (CART) using Gini Criteria

# In[23]:


model_gini = DecisionTreeClassifier(criterion='gini')


# In[24]:


model_gini.fit(x_train, y_train)


# In[25]:


model_gini.get_n_leaves()


# In[26]:


preds = model_gini.predict(x_test) # predicting on test data set 
pd.Series(preds).value_counts() # getting the count of each category 


# In[27]:


preds


# In[28]:


# Accuracy 
np.mean(preds==y_test)


# In[29]:


print(classification_report(preds,y_test))


# # Building Decision Tree Regression

# In[30]:


from sklearn.tree import DecisionTreeRegressor


# In[31]:


model_R = DecisionTreeRegressor()
model_R.fit(x_train, y_train)


# In[32]:


preds = model_R.predict(x_test) 


# In[33]:


np.mean(preds==y_test)


# # Plot Tree Diagram

# In[34]:


# Decision Tree Classifier using Entropy Criteria
fig = plt.figure(figsize=(25,20))
fig = tree.plot_tree(model)


# In[35]:


# Decision Tree Classifier (CART) using Gini Criteria
fig = plt.figure(figsize=(25,20))
fig = tree.plot_tree(model_gini)


# In[36]:


# Decision Tree Regression
fig = plt.figure(figsize=(25,20))
fig = tree.plot_tree(model_R)

