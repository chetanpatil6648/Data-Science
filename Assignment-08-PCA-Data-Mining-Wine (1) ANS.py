#!/usr/bin/env python
# coding: utf-8

# In[1]:


# PCA Assignment


# In[2]:


# Import libraries
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale 
import seaborn as sns


# In[3]:


# Load data sets
wine = pd.read_csv("\\Users\\Rohit\\OneDrive\\Desktop\\assingment file\\wine.csv")


# In[4]:


wine


# In[5]:


wine['Type'].value_counts()


# In[6]:


wine2=wine.iloc[:,1:]
wine2


# In[7]:


wine2.info()


# In[8]:


# Converting the data into numpy array
wine_ary = wine2.values
wine_ary


# In[9]:


# Normalizing the numerical data 
wine_norm=scale(wine_ary)
wine_norm


# # PCA Implementation

# In[10]:


pca = PCA(n_components=13)


wine_pca=pca.fit_transform(wine_norm)
wine_pca


# In[11]:


# PCA Components matrix or covariance Matrix
pca.components_


# In[12]:


# The amount of variance that each PCA has
var=pca.explained_variance_ratio_
var


# In[13]:


# Cummulative variance of each PCA
var1=np.cumsum(np.round(var,4)*100)
var1


# In[14]:


# Variance plot for PCA components obtained 
plt.plot(var1,color='magenta')


# In[15]:


# Final Dataframe
final_df=pd.concat([wine['Type'],pd.DataFrame(wine_pca[:,0:3],columns=['PC1','PC2','PC3'])],axis=1)
final_df


# In[16]:


# Visualization of PCAs
fig=plt.figure(figsize=(16,12))
sns.scatterplot(data=final_df)


# # Checking with other Clustering Algorithms
# # 1. Hierarchical Clustering

# In[17]:


# Import Libraries
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize


# In[18]:


# As we already have normalized data, create Dendrograms
plt.figure(figsize=(10,8))
dendrogram=sch.dendrogram(sch.linkage(wine_norm,'complete'))


# In[19]:


# Create Clusters (y)
hclusters=AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='complete')
hclusters


# In[20]:


y = pd.DataFrame(hclusters.fit_predict(wine_norm),columns=['clustersid'])
y['clustersid'].value_counts()


# In[21]:


# Adding clusters to dataset
wine3=wine.copy()
wine3['clustersid']=hclusters.labels_
wine3


# # 2. K-Means Clustering

# In[22]:


# Import Libraries
from sklearn.cluster import KMeans


# In[23]:


# As we already have normalized data
# Use Elbow Graph to find optimum number of  clusters (K value) from K values range
# The K-means algorithm aims to choose centroids that minimise the inertia, or within-cluster sum-of-squares criterion WCSS 
# random state can be anything from 0 to 42, but the same number to be used everytime,so that the results don't change. 


# In[24]:


# within-cluster sum-of-squares criterion 
wcss=[]
for i in range (1,6):
    kmeans=KMeans(n_clusters=i,random_state=2)
    kmeans.fit(wine_norm)
    wcss.append(kmeans.inertia_)


# In[25]:


# Plot K values range vs WCSS to get Elbow graph for choosing K (no. of clusters)
plt.plot(range(1,6),wcss)
plt.title('Elbow Graph')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# # Build Cluster algorithm using K=3

# In[26]:


# Cluster algorithm using K=3
clusters3=KMeans(3,random_state=30).fit(wine_norm)
clusters3


# In[27]:


clusters3.labels_


# In[28]:


# Assign clusters to the data set
wine4=wine.copy()
wine4['clusters3id']=clusters3.labels_
wine4


# In[29]:


wine4['clusters3id'].value_counts()

