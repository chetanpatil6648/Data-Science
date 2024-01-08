#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Hierarchical, K means clustering and DBSCAN of Airlines data sets Assignment


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
airline = pd.read_csv("\\Users\\Rohit\\OneDrive\\Desktop\\assingment file\\EastWestAirlines.csv")


# In[4]:


airline


# # EDA

# In[5]:


airline.info()


# In[6]:


# data has clear


# In[7]:


airline2=airline.drop(['ID#'],axis=1)
airline2


# In[8]:


# Normalization function 
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)


# In[9]:


# Normalized data frame (considering the numerical part of data)
airline2_norm = norm_func(airline2)
airline2_norm


# In[10]:


# Create Dendrograms
plt.figure(figsize=(10, 7))  
dendograms=sch.dendrogram(sch.linkage(airline2_norm,'complete'))


# In[11]:


# Create Clusters (y)
hclusters=AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='complete')
hclusters


# In[12]:


y=pd.DataFrame(hclusters.fit_predict(airline2_norm),columns=['clustersid'])
y['clustersid'].value_counts()


# In[13]:


# Adding clusters to dataset
airline2['clustersid']=hclusters.labels_
airline2


# In[14]:


airline2.groupby('clustersid').agg(['mean']).reset_index()


# In[15]:


# Plot Clusters
plt.figure(figsize=(10, 7))  
plt.scatter(airline2['clustersid'],airline2['Balance'], c=hclusters.labels_) 


# # K means clustering

# In[16]:


# Import Libraries
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans


# In[17]:


airline2_norm    # Normalies data sets


# In[20]:


# Selecting 5 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters=5)
model.fit(airline2_norm)
model.labels_                    # getting the labels of clusters assigned to each row 


# In[23]:


md = pd.Series(model.labels_)  # converting numpy array into pandas series object 
airline2['clusters'] = md          # creating a  new column and assigning it to new column 
airline2


# In[24]:


airline2.groupby(airline2.clusters).mean()


# In[34]:


# Plot Clusters
plt.figure(figsize=(10, 7))  
plt.scatter(airline2['clusters'],airline2['Balance'], c=model.labels_) 


# # DBSCAN

# In[35]:


# Import Libraries
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN


# In[36]:


airline2_norm    # Normalies data sets


# In[37]:


# creating a clusters
dbscan = DBSCAN(eps=0.8, min_samples=15)
dbscan.fit(airline2_norm)


# In[38]:


#Noisy samples are given the label -1.
dbscan.labels_


# In[39]:


ml=pd.DataFrame(dbscan.labels_,columns=['cluster'])
pd.concat([airline2_norm,ml],axis=1)

