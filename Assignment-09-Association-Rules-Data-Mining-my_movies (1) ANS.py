#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Assignment


# In[2]:


# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori,association_rules
from mlxtend.preprocessing import TransactionEncoder


# In[3]:


# Load dataset
movie = pd.read_csv("\\Users\\Rohit\\OneDrive\\Desktop\\assingment file\\my_movies.csv")


# In[4]:


movie


# In[6]:


movie.info()


# In[7]:


movie_2 = movie.iloc[:,5:]
movie_2


# # Apriori Algorithm
# ## 1. Association rules with 10% Support and 70% confidence

# In[9]:


# with 10% support
frequent_itemsets=apriori(movie_2,min_support=0.1,use_colnames=True)
frequent_itemsets


# In[10]:


# 70% confidence
rules=association_rules(frequent_itemsets,metric='lift',min_threshold=0.7)
rules


# In[11]:


# Lift Ratio > 1 is a good influential rule in selecting the associated transactions
rules[rules.lift>1]


# In[12]:


# visualization of obtained rule
plt.scatter(rules.support,rules.confidence)
plt.xlabel('support')
plt.ylabel('confidence')
plt.show()


# ## 1. Association rules with 5% Support and 90% confidence

# In[13]:


# with 5% support
frequent_itemsets2=apriori(movie_2,min_support=0.05,use_colnames=True)
frequent_itemsets2


# In[14]:


# 90% confidence
rules2=association_rules(frequent_itemsets2,metric='lift',min_threshold=0.9)
rules2


# In[15]:


# Lift Ratio > 1 is a good influential rule in selecting the associated transactions
rules2[rules2.lift>1]


# In[16]:


# visualization of obtained rule
plt.scatter(rules2.support,rules2.confidence)
plt.xlabel('support')
plt.ylabel('confidence')
plt.show()


# ## 1. Association rules with 30% Support and 50% confidence

# In[17]:


# with 5% support
frequent_itemsets3=apriori(movie_2,min_support=0.3,use_colnames=True)
frequent_itemsets3


# In[18]:


# 50% confidence
rules3=association_rules(frequent_itemsets3,metric='lift',min_threshold=0.5)
rules3


# In[19]:


# Lift Ratio > 1 is a good influential rule in selecting the associated transactions
rules3[rules3.lift>1]


# In[20]:


# visualization of obtained rule
plt.scatter(rules3.support,rules3.confidence)
plt.xlabel('support')
plt.ylabel('confidence')
plt.show()

