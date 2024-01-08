#!/usr/bin/env python
# coding: utf-8

# In[1]:


#  Hierarchical, K means clustering and DBSCAN of Crime data sets Assignment


# # Hierarchical clustering

# In[2]:


# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering


# In[3]:


# Load data sets
crime = pd.read_csv("\\Users\\Rohit\\OneDrive\\Desktop\\assingment file\\crime_data.csv")


# In[4]:


crime.head()


# In[5]:


crime.drop(['Unnamed: 0'],axis=1,inplace=True)
crime.head()


# In[7]:


crime.info()     # Data has clear


# In[8]:


# Normalization function 
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)


# In[9]:


# Normalized data frame (considering the numerical part of data)
crime_norm = norm_func(crime)
crime_norm


# In[10]:


# Create Dendrograms
plt.figure(figsize=(10, 7))  
dendograms=sch.dendrogram(sch.linkage(crime_norm,'complete'))


# In[11]:


# Create Clusters (y)
hclusters=AgglomerativeClustering(n_clusters=4,affinity='euclidean',linkage='complete')
hclusters


# In[12]:


y=pd.DataFrame(hclusters.fit_predict(crime_norm),columns=['clustersid'])
y['clustersid'].value_counts()


# In[13]:


# Adding clusters to dataset
crime['clustersid']=hclusters.labels_
crime


# In[14]:


crime.groupby('clustersid').agg(['mean']).reset_index()


# In[15]:


# Plot Clusters
plt.figure(figsize=(10, 7))  
plt.scatter(crime['clustersid'],crime['Murder'], c=hclusters.labels_) 


# # K means clustering

# In[16]:


# Import Libraries
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans


# In[27]:


# Load data sets
crime = pd.read_csv("\\Users\\Rohit\\OneDrive\\Desktop\\assingment file\\crime_data.csv")


# In[28]:


crime.drop(['Unnamed: 0'],axis=1,inplace=True)
crime.head()


# In[29]:


# Normalization function 
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)


# In[30]:


# Normalized data frame (considering the numerical part of data)
crime_norm = norm_func(crime)


# In[31]:


crime_norm


# In[32]:


# Selecting 4 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters=4)
model.fit(crime_norm)
model.labels_                    # getting the labels of clusters assigned to each row 


# In[33]:


md = pd.Series(model.labels_)  # converting numpy array into pandas series object 
crime2['clusters'] = md          # creating a  new column and assigning it to new column 
crime2


# In[34]:


crime2.groupby(crime2.clusters).mean()


# In[35]:


# Plot Clusters
plt.figure(figsize=(10, 7))  
plt.scatter(crime2['clusters'],crime2['Murder'], c=model.labels_)


# # DBSCAN

# In[22]:


# Import Libraries
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN


# In[23]:


crime_norm.head()


# In[24]:


# DBSCAN Clustering
dbscan=DBSCAN(eps=1,min_samples=4)
dbscan.fit(crime_norm)


# In[25]:


#Noisy samples are given the label -1.
dbscan.labels_


# In[37]:


ml=pd.DataFrame(dbscan.labels_,columns=['cluster'])
pd.concat([crime_norm,ml],axis=1)

