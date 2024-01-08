#!/usr/bin/env python
# coding: utf-8

#  Consider only the below columns and prepare a prediction model for 50_startups data. Profit

# In[1]:


# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.graphics.regressionplots import influence_plot


# In[2]:


data = pd.read_csv("\\Users\\Rohit\\OneDrive\\Desktop\\assingment file\\50_Startups.csv")


# In[3]:


data


# In[4]:


data.info()


# In[5]:


data1=data.rename({'R&D Spend':'RDS','Administration':'ADMS','Marketing Spend':'MKTS'},axis=1)
data1


# In[6]:


data1[data1.duplicated()]


# In[7]:


# NO duplicate value


# In[8]:


data1.describe()


# In[9]:


data1.corr()


# In[10]:


sns.set_style(style='darkgrid')
sns.pairplot(data1)


# # Model Building

# In[11]:


model = smf.ols('Profit~RDS+ADMS+MKTS',data=data1).fit()


# # Model Testing

# In[12]:


model.params


# In[13]:


print(model.tvalues,'\n',model.pvalues)


# In[14]:


(model.rsquared,model.rsquared_adj)


# In[15]:


# Build SLR and MLR models for insignificant variables 'ADMS' and 'MKTS'
# Also find their tvalues and pvalues


# In[16]:


ml_ADMS = smf.ols("Profit~ADMS",data=data1).fit()
print(ml_ADMS.tvalues,'/n',ml_ADMS.pvalues)         # ADMS has in-significant pvalue


# In[17]:


ml_MKTS = smf.ols("Profit~MKTS",data=data1).fit()
print(ml_MKTS.tvalues,'/n',ml_MKTS.pvalues)         # MKTS has significant pvalue


# In[18]:


ml_AM = smf.ols("Profit~ADMS+MKTS",data=data1).fit()
print(ml_AM.tvalues,'/n',ml_AM.pvalues)         # AM has significant pvalue


# # Model Validation Techniques
# Two Techniques: 1. Collinearity Check & 2. Residual Analysis

# In[19]:


# 1) Collinearity Problem Check
# Calculate VIF = 1/(1-Rsquare) for all independent variables
rsq_RDS = smf.ols('RDS~ADMS+MKTS',data=data1).fit().rsquared
vif_RDS = 1/(1-rsq_RDS)
rsq_ADMS = smf.ols('ADMS~RDS+MKTS',data=data1).fit().rsquared
vif_ADMS = 1/(1-rsq_ADMS)
rsq_MKTS = smf.ols('MKTS~RDS+ADMS',data=data1).fit().rsquared
vif_MKTS = 1/(1-rsq_MKTS)
# Putting the values in Dataframe format
d1={'Variables':['RDS','ADMS','MKTS'],
    'Vif':[vif_RDS,vif_ADMS,vif_MKTS]}
Vif_df=pd.DataFrame(d1)
Vif_df


# None variable has VIF>20, No Collinearity, so consider all varaibles in Regression equation

# In[20]:


# 2) Residual Analysis
# Test for Normality of Residuals (Q-Q Plot) using residual model (model.resid)
sm.qqplot(model.resid,line='q') # 'q' - A line is fit through the quartiles # line = '45'- to draw the 45-degree diagonal line
plt.title("Normal Q-Q plot of residuals")
plt.show()


# In[21]:


list(np.where(model.resid<-30000))


# In[22]:


# Test for errors or Residuals Vs Regressors or independent 'x' variables or predictors 
# using Residual Regression Plots code graphics.plot_regress_exog(model,'x',fig)    # exog = x-variable & endog = y-variable


# In[23]:


fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'RDS',fig=fig)
plt.show()


# In[24]:


fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'ADMS',fig=fig)
plt.show()


# In[25]:


fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'MKTS',fig=fig)
plt.show()


# # Model Deletion Diagnostics (checking Outliers or Influencers)
# Two Techniques : 1. Cook's Distance & 2. Leverage value

# In[26]:


# 1. Cook's Distance: If Cook's distance > 1, then it's an outlier
# Get influencers using cook's distance
(c,_)=model.get_influence().cooks_distance
c


# In[27]:


# Plot the influencers using the stem plot
fig=plt.figure(figsize=(20,7))
plt.stem(np.arange(len(data1)),np.round(c,3))
plt.xlabel('Row Index')
plt.ylabel('Cooks Distance')
plt.show()


# In[28]:


# Index and value of influencer where C>0.5
np.argmax(c) , np.max(c)


# In[29]:


# 2. Leverage Value using High Influence Points : Points beyond Leverage_cutoff value are influencers
fig,ax=plt.subplots(figsize=(20,20))
fig=influence_plot(model,ax = ax)


# In[30]:


# Leverage Cuttoff Value = 3*(k+1)/n ; k = no.of features/columns & n = no. of datapoints
k=data1.shape[1]
n=data1.shape[0]
leverage_cutoff = (3*(k+1))/n
leverage_cutoff


# In[31]:


data1[data1.index.isin([49])]


# # Improving the Model

# In[32]:


# Creating a copy of data so that original dataset is not affected
data2 =data1.copy()
data2


# In[33]:


# Discard the data points which are influencers and reassign the row number (reset_index(drop=True))
data2 = data2.drop(data2.index[[49]],axis=0).reset_index(drop=True)
data2


# # Model Deletion Diagnostics and Final Model

# In[34]:


while model.rsquared < 0.99:
    for c in [np.max(c)>0.5]:
        model=smf.ols('Profit~RDS+ADMS+MKTS',data=data2).fit()
        (c,_)=model.get_influence().cooks_distance
        c
        np.argmax(c) , np.max(c)
        data2=data2.drop(data2.index[[np.argmax(c)]],axis=0).reset_index(drop=True)
        data2
    else:
        final_model=smf.ols('Profit~RDS+ADMS+MKTS',data=data2).fit()
        final_model.rsquared , final_model.aic
        print("Thus model accuracy is improved to",final_model.rsquared)


# In[35]:


final_model.rsquared


# In[36]:


data2


# 
# # Model Predictions

# In[37]:


# say New data for prediction is
new_data=pd.DataFrame({'RDS':70000,"ADMS":90000,"MKTS":140000},index=[0])
new_data


# In[38]:


# Manual Prediction of Price
final_model.predict(new_data)


# In[39]:


# Automatic Prediction of Price with 90.02% accurcy
pred_y=final_model.predict(data2)
pred_y


# # table containing R^2 value for each prepared model

# In[40]:


d2={'Prep_Models':['Model','Final_Model'],'Rsquared':[model.rsquared,final_model.rsquared]}
table=pd.DataFrame(d2)
table

