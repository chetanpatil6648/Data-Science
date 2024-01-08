#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Hypothesis assignment


# In[2]:


#Import libraries
import numpy as np
import scipy as sp
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.proportion import proportions_ztest


# In[3]:


# Load data set
data = pd.read_csv("\\Users\\Rohit\\OneDrive\\Desktop\\assingment file\\Cutlets.csv")


# In[4]:


data


# In[5]:


# data analysis


# In[6]:


data.shape


# In[7]:


data.info()


# In[8]:


data.describe(include='all')


# In[9]:


unit_a = data["Unit A"].mean()


# In[10]:


unit_a


# In[11]:


unit_b = data["Unit B"].mean()


# In[12]:


print('Unit A mean is',unit_a,'\n','Unit B mean is',unit_b)


# In[13]:


print('Unit A > Unit B:',unit_a>unit_b)


# In[14]:


#visualization


# In[15]:


sns.distplot(data['Unit A'])
sns.distplot(data['Unit B'])
plt.legend(['Unit A','Unit B'])


# In[16]:


sns.boxplot(data=[data['Unit A'],data['Unit B']])
plt.legend(['Unit A','Unit B'])


# In[17]:


alpha=0.05
UnitA = pd.DataFrame(data['Unit A'])
UnitA


# In[18]:


UnitB = pd.DataFrame(data['Unit B'])
UnitB


# In[19]:


print(UnitA,UnitB)


# In[20]:


tStat,pValue =sp.stats.ttest_ind(UnitA,UnitB)


# In[21]:


print("P-Value:{0} T-Statistic:{1}".format(pValue,tStat))


# In[22]:


sp.stats.ttest_ind(UnitA,UnitB)


# In[23]:


if pValue <0.05:
  print('we reject null hypothesis')
else:
  print('we accept null hypothesis')


# In[24]:


#Inference : No significant difference in diameter of Unit A and Unit B

