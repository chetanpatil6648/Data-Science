#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Hypothesis Assignment


# In[2]:


# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import stats
from scipy.stats import norm
from scipy.stats import chi2_contingency


# In[3]:


# Load data set
buyer_ratio = pd.read_csv("\\Users\\Rohit\\OneDrive\\Desktop\\assingment file\\BuyerRatio.csv")


# In[4]:


buyer_ratio


# In[5]:


#Assume Null Hypothesis as Ho: Independence of categorical variables (male-female buyer rations are similar across regions (does not vary and are not related) Thus Alternate Hypothesis as Ha: Dependence of categorical variables (male-female buyer rations are NOT similar across regions (does vary and somewhat/significantly related)


# In[6]:


obs = buyer_ratio.iloc[:,1:]


# In[7]:


obs


# In[8]:


obs = obs.values


# In[9]:


obs


# In[10]:


chi2_contingency(obs)


# In[11]:


# Compare p_value with α = 0.05


# In[12]:


pValue = 0.66030


# In[13]:


if pValue <0.05:
  print('All proportions are equal')
else:
  print('Not all proportions are equal')


# In[14]:


#Inference: As (p-value = 0.6603) > (α = 0.05); Accept the Null Hypothesis i.e. Independence of categorical variables Thus, male-female buyer rations are similar across regions and are not related

