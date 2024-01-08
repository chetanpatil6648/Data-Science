#!/usr/bin/env python
# coding: utf-8

# Q1) Delivery_time -> Predict delivery time using sorting time Build a simple linear regression model by performing EDA and do necessary transformations and select the best model using R or Python.# 

# In[1]:


# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf


# In[2]:


data = pd.read_csv("\\Users\\Rohit\\OneDrive\\Desktop\\assingment file\\delivery_time.csv")


# In[3]:


data


# # EDA & Data Visualization

# In[4]:


data.info()


# In[5]:


sns.distplot(data['Delivery Time'])


# In[6]:


sns.distplot(data['Sorting Time'])


# In[7]:


# Renaming Columns
data=data.rename({'Delivery Time':'delivery_time', 'Sorting Time':'sorting_time'},axis=1)
data


# # Correlation Analysis

# In[8]:


data.corr()


# # Model Building

# In[9]:


model = smf.ols('delivery_time~sorting_time',data=data).fit()


# In[10]:


sns.regplot(x='sorting_time',y='delivery_time',data=data)


# # Model Testing

# In[11]:


model.params       # Finding Coefficient parameters


# In[12]:


# Finding tvalues and pvalues
(model.tvalues,'\n',model.pvalues)


# In[13]:


# Finding Rsquared Values
model.rsquared , model.rsquared_adj


# # Model Predictions

# In[14]:


# Manual prediction for say sorting time 5
delivery_time = (6.582734) + (1.649020)*(5)
delivery_time


# In[15]:


# Automatic Prediction for say sorting time 5, 8
new_data=pd.Series([5,8])
new_data


# In[16]:


data_pred=pd.DataFrame(new_data,columns=['sorting_time'])
data_pred


# In[17]:


model.predict(data_pred)

