#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Hypothesis Assignment


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


# Load deta set
data = pd.read_csv("\\Users\\Rohit\\OneDrive\\Desktop\\assingment file\\LabTAT.csv")


# In[4]:


data.head()


# In[5]:


# data analysis
data.info()


# In[6]:


data.describe()


# In[7]:


Laboratory_1 = data['Laboratory 1'].mean()
print('Laboratory 1 mean =',Laboratory_1)


# In[8]:


Laboratory_2 = data['Laboratory 2'].mean()
print('Laboratory 2 mean =',Laboratory_2)


# In[9]:


Laboratory_3 = data['Laboratory 3'].mean()
print('Laboratory 3 mean =',Laboratory_3)


# In[10]:


Laboratory_4 = data['Laboratory 4'].mean()
print('Laboratory 4 mean =',Laboratory_4)


# In[11]:


print('Laboratory_1 > Laboratory_2 = ',Laboratory_1 > Laboratory_2)
print('Laboratory_2 > Laboratory_3 = ',Laboratory_2 > Laboratory_3)
print('Laboratory_3 > Laboratory_4 = ',Laboratory_3 > Laboratory_4)
print('Laboratory_4 > Laboratory_1 = ',Laboratory_4 > Laboratory_1)


# In[12]:


#The Null and Alternative Hypothesis

#There are no significant differences between the groups' mean Lab values. H0:μ1=μ2=μ3=μ4

#There is a significant difference between the groups' mean Lab values. Ha:μ1≠μ2≠μ3≠μ4


# In[13]:


# Visualization


# In[14]:


sns.distplot(data['Laboratory 1'])
sns.distplot(data['Laboratory 2'])
sns.distplot(data['Laboratory 3'])
sns.distplot(data['Laboratory 4'])
plt.legend(['Laboratory 1','Laboratory 2','Laboratory 3','Laboratory 4'])


# In[15]:


plt.figure(figsize=(15,10))
sns.boxplot(data=[data['Laboratory 1'],data['Laboratory 2'],data['Laboratory 3'],data['Laboratory 4']])
plt.legend(['Laboratory 1','Laboratory 2','Laboratory 3','Laboratory 4'])


# In[16]:


#Hypithesis Testing


# In[17]:


alpha=0.05
Lab1=pd.DataFrame(data['Laboratory 1'])
Lab2=pd.DataFrame(data['Laboratory 2'])
Lab3=pd.DataFrame(data['Laboratory 3'])
Lab4=pd.DataFrame(data['Laboratory 4'])


# In[18]:


print(Lab1,Lab2,Lab3,Lab4)


# In[19]:


tStat, pvalue = sp.stats.f_oneway(Lab1,Lab2,Lab3,Lab4)


# In[20]:


print("P-Value:{0} T-Statistic:{1}".format(pvalue,tStat))


# In[21]:


if pvalue < 0.05:
  print('we reject null hypothesis')
else:
  print('we accept null hypothesis')


# In[22]:


#Inference : There is no significant difference in the average TAT for all the labs.

