#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Assignments


# In[2]:


#import libraries
from pandas import read_csv
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier


# In[3]:


# Load datasets
zoo = read_csv("\\Users\\Rohit\\Downloads\\Zoo.csv")
zoo


# In[4]:


zoo.info()


# # Preprocessing

# In[5]:


from sklearn import preprocessing 
label_encoder = preprocessing.LabelEncoder()
zoo["animal name"] = label_encoder.fit_transform(zoo["animal name"])


# In[6]:


zoo.head()


# In[7]:


array = zoo.values
X = array[:, 1:17]
X


# In[8]:


Y = array[:, -1]
Y


# In[9]:


kfold = KFold(n_splits=4)


# In[10]:


model = KNeighborsClassifier(n_neighbors=13)
results = cross_val_score(model, X, Y, cv=kfold)


# In[11]:


print(results.mean())


# # Grid Search for Algorithm Tuning

# In[12]:


import numpy
from pandas import read_csv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


# In[13]:


n_neighbors1 = numpy.array(range(1,40))
param_grid = dict(n_neighbors=n_neighbors1)


# In[14]:


model = KNeighborsClassifier()
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid.fit(X, Y)


# In[15]:


print(grid.best_score_)


# In[16]:


print(grid.best_params_)


# # visualize the CV results

# In[17]:


import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
# choose k between 1 to 70
k_range = range(1, 70)
k_scores = []
# use iteration to caclulator different k in models, then return the average accuracy based on the cross validation
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, Y, cv=4)
    k_scores.append(scores.mean())
# plot to see clearly
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()

