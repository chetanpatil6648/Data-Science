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


# load data sets
book = pd.read_csv("\\Users\\Rohit\\OneDrive\\Desktop\\assingment file\\book.csv")


# In[4]:


book.head()


# In[5]:


book.info()    # data has clear


# # Apriori Algorithm
# ## 1. Association rules with 10% Support & 70% confidence

# In[6]:


# With 10% Support
frequent_itemsets = apriori(book,min_support=0.1,use_colnames=True)
frequent_itemsets


# In[8]:


# With 70% confidence 
rules = association_rules(frequent_itemsets,metric='lift',min_threshold=0.7)
rules


# In[9]:


#  An leverage value of 0 indicates independence. Range will be [-1 1]
# High conviction value means that the consequent is highly depending on the antecedent and range [0 inf]


# In[11]:


rules.sort_values('lift',ascending=False)


# In[12]:


# Lift Ratio > 1 is a good influential rule in selecting the associated transactions
rules[rules.lift>1]


# In[13]:


# visualization of obtained rule
plt.scatter(rules.support,rules.confidence)
plt.xlabel('support')
plt.ylabel('confidence')
plt.show()


# ## 2. Association rules with 20% Support & 60% confidence

# In[14]:


# With 20% Support
frequent_itemsets2 = apriori(book,min_support=0.2,use_colnames=True)
frequent_itemsets2


# In[15]:


# With 60% confidence 
rules2 = association_rules(frequent_itemsets2,metric='lift',min_threshold=0.6)
rules2


# In[16]:


# visualization of obtained rule
plt.scatter(rules2.support,rules2.confidence)
plt.xlabel('support')
plt.ylabel('confidence')
plt.show()


# ## 3. Association rules with 5% Support and 80% confidence

# In[17]:


# With 5% Support
frequent_itemsets3=apriori(book,min_support=0.05,use_colnames=True)
frequent_itemsets3


# In[18]:


# With 80% confidence
rules3=association_rules(frequent_itemsets3,metric='lift',min_threshold=0.8)
rules3


# In[19]:


rules3[rules3.lift>1]


# In[20]:


# visualization of obtained rule
plt.scatter(rules3.support,rules3.confidence)
plt.xlabel('support')
plt.ylabel('confidence')
plt.show()

