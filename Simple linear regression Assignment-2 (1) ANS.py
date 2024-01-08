#!/usr/bin/env python
# coding: utf-8

# Q2) Salary_hike -> Build a prediction model for Salary_hike Build a simple linear regression model by performing EDA and do necessary transformations and select the best model using R or Python.

# In[1]:


# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf


# In[2]:


data = pd.read_csv("\\Users\\Rohit\\OneDrive\\Desktop\\assingment file\\Salary_Data.csv")


# In[3]:


data


# # EDA & Data Visualization

# In[4]:


data.info()


# In[5]:


sns.distplot(data['YearsExperience'])


# In[6]:


sns.distplot(data['Salary'])


# In[7]:


sns.distplot(data['YearsExperience'])
sns.distplot(data['Salary'])
plt.legend(['YearsExperience','Salary'])


# In[8]:


# Renaming Columns
data=data.rename({'Salary':'Salary_hike'},axis=1)


# # Correlation Analysis

# In[9]:


data.corr()


# # Model Building

# In[10]:


model = smf.ols('Salary_hike~YearsExperience',data=data).fit()


# In[11]:


sns.regplot(x='YearsExperience',y='Salary_hike',data=data)


# # Model Testing

# In[12]:


model.params       # Finding Coefficient parameters


# In[13]:


# Finding tvalues and pvalues
(model.tvalues,'\n',model.pvalues)


# In[14]:


# Finding Rsquared Values
model.rsquared , model.rsquared_adj


# # Model Predictions

# In[15]:


# Manual prediction for say 3 Years Experience
Salary_hike = (25792.200199) + (9449.962321)*(3)
Salary_hike


# In[16]:


# Automatic Prediction for say 3 & 5 Years Experience


# In[17]:


new_data=pd.Series([3,5])
new_data


# In[18]:


data_pred=pd.DataFrame(new_data,columns=['YearsExperience'])
data_pred


# In[19]:


model.predict(data_pred)

