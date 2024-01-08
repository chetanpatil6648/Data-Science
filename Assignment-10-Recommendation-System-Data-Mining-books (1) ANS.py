#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Import Liabraries
import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine,correlation


# In[4]:


books = pd.read_csv("\\Users\\Rohit\\OneDrive\\Desktop\\assingment file\\book.csv")
books


# In[6]:


books_2 = books.iloc[:,1:]
books_2


# In[7]:


books_2.sort_values(['User.ID'])


# In[9]:


# number of unique users in the dataset
len(books_2['User.ID'].unique())


# In[11]:


# number of unique books in the dataset
len(books_2['Book.Title'].unique())


# In[12]:


books_3 = books_2.pivot_table(index='User.ID',
                                 columns='Book.Title',
                                 values='Book.Rating').reset_index(drop=True)


# In[13]:


books_3


# In[15]:


# Replacing the index values by unique user Ids
books_3.index = books_2['User.ID'].unique()
books_3


# In[17]:


# Impute those NaNs with 0 values
books_3.fillna(0,inplace=True)
books_3


# In[18]:


# Calculating Cosine Similarity between Users on array data
user_sim = 1-pairwise_distances(books_3.values,metric='cosine')
user_sim


# In[19]:


# Store the results in a dataframe format
user_sim2=pd.DataFrame(user_sim)
user_sim2


# In[21]:


# Set the index and column names to user ids 
user_sim2.index=books_2['User.ID'].unique()
user_sim2.columns=books_2['User.ID'].unique()
user_sim2


# In[22]:


# Nullifying diagonal values
np.fill_diagonal(user_sim,0)
user_sim2


# In[23]:


# Most Similar Users
user_sim2.idxmax(axis=1)


# In[25]:


# extract the books which userId 162107 & 276726 have watched
books_2[(books_2['User.ID']==162107) | (books_2['User.ID']==276726)]


# In[26]:


# extract the books which userId 276729 & 276726 have watched
books_2[(books_2['User.ID']==276729) | (books_2['User.ID']==276726)]


# In[27]:


user_1=books_2[(books_2['User.ID']==276729)]
user_2=books_2[(books_2['User.ID']==276726)]


# In[28]:


user_1['Book.Title']


# In[29]:


user_2['Book.Title']


# In[30]:


pd.merge(user_1,user_2,on='Book.Title',how='outer')

