#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Hypothesis Assignment


# In[2]:


# Impot libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import stats
from scipy.stats import norm
from scipy.stats import chi2_contingency


# In[3]:


customer_order = pd.read_csv("\\Users\\Rohit\\OneDrive\\Desktop\\assingment file\\Costomer+OrderForm.csv")


# In[4]:


customer_order


# In[5]:


customer_order.Phillippines.value_counts()


# In[6]:


customer_order.Indonesia.value_counts()


# In[7]:


customer_order.Malta.value_counts()


# In[8]:


customer_order.India.value_counts()


# In[9]:


obs = np.array([[271,267,269,280],[29,33,31,20]])


# In[10]:


obs


# In[11]:


# Chi2 contengency independence test
chi2_contingency(obs)   # o/p is (Chi2 stats value, p_value, df, expected obsvations)


# In[12]:


pValue =  0.2771


# In[13]:


#Assume Null Hypothesis as Ho: Independence of categorical variables (customer order forms defective % does not varies by centre) Thus, Alternative hypothesis as Ha Dependence of categorical variables (customer order forms defective % varies by centre)


# In[14]:


# Compare p_value with α = 0.05


# In[15]:


if pValue< 0.05:
  print('we reject null hypothesis')
else:
  print('we accept null hypothesis')


# In[16]:


# Inference: As (p_value = 0.2771) > (α = 0.05); Accept Null Hypthesis i.e. Independence of categorical variables Thus, customer order forms defective % does not varies by centre

