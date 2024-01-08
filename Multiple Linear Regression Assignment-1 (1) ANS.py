#!/usr/bin/env python
# coding: utf-8

# Consider only the below columns and prepare a prediction model for predicting Price. Corolla 

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


# Load data set
toyota = pd.read_csv("\\Users\\Rohit\\OneDrive\\Desktop\\assingment file\\ToyotaCorolla (1).csv") 


# In[3]:


toyota.head()


# In[4]:


toyota2=pd.concat([toyota.iloc[:,2:4],toyota.iloc[:,6:7],toyota.iloc[:,8:9],toyota.iloc[:,12:14],toyota.iloc[:,15:18]],axis=1)
toyota2


# In[5]:


toyota2.info()


# In[6]:


toyota3 = toyota2.rename({'Age_08_04':'Age','cc':'CC','Quarterly_Tax':'QT'},axis=1)


# In[7]:


toyota3


# In[8]:


toyota4 = toyota3.drop_duplicates().reset_index(drop=True)


# In[9]:


toyota4


# In[10]:


toyota4.describe()


# In[11]:


toyota4.corr()


# In[12]:


sns.set_style(style='darkgrid')
sns.pairplot(toyota4)


# # Model Building

# In[13]:


model = smf.ols("Price~Age+KM+HP+CC+Doors+Gears+QT+Weight",data=toyota4).fit()


# # Model Testing

# In[14]:


model.params


# In[15]:


print(model.tvalues,'/n',model.pvalues)


# In[16]:


(model.rsquared,model.rsquared_adj)


# In[17]:


# Build SLR and MLR models for insignificant variables 'CC' and 'Doors'
# Also find their tvalues and pvalues


# In[18]:


ml_cc = smf.ols("Price~CC",data=toyota4).fit()
print(ml_cc.tvalues,'/n',ml_cc.pvalues)         # CC has significant pvalue


# In[19]:


ml_doors = smf.ols("Price~Doors",data=toyota4).fit()
print(ml_doors.tvalues,'/n',ml_doors.pvalues)             # Doors has significant pvalue


# In[20]:


ml_cd = smf.ols("Price~CC+Doors",data=toyota4).fit()
print(ml_cd.tvalues,'/n',ml_cd.pvalues)                  # CC & Doors have significant pvalue


# # Model Validation Techniques
# Two Techniques: 1. Collinearity Check & 2. Residual Analysis

# In[21]:


# 1) Collinearity Problem Check
# Calculate VIF = 1/(1-Rsquare) for all independent variables

rsq_age=smf.ols('Age~KM+HP+CC+Doors+Gears+QT+Weight',data=toyota4).fit().rsquared
vif_age=1/(1-rsq_age)

rsq_KM=smf.ols('KM~Age+HP+CC+Doors+Gears+QT+Weight',data=toyota4).fit().rsquared
vif_KM=1/(1-rsq_KM)

rsq_HP=smf.ols('HP~Age+KM+CC+Doors+Gears+QT+Weight',data=toyota4).fit().rsquared
vif_HP=1/(1-rsq_HP)

rsq_CC=smf.ols('CC~Age+KM+HP+Doors+Gears+QT+Weight',data=toyota4).fit().rsquared
vif_CC=1/(1-rsq_CC)

rsq_DR=smf.ols('Doors~Age+KM+HP+CC+Gears+QT+Weight',data=toyota4).fit().rsquared
vif_DR=1/(1-rsq_DR)

rsq_GR=smf.ols('Gears~Age+KM+HP+CC+Doors+QT+Weight',data=toyota4).fit().rsquared
vif_GR=1/(1-rsq_GR)

rsq_QT=smf.ols('QT~Age+KM+HP+CC+Doors+Gears+Weight',data=toyota4).fit().rsquared
vif_QT=1/(1-rsq_QT)

rsq_WT=smf.ols('Weight~Age+KM+HP+CC+Doors+Gears+QT',data=toyota4).fit().rsquared
vif_WT=1/(1-rsq_WT)

# Putting the values in Dataframe format
d1={'Variables':['Age','KM','HP','CC','Doors','Gears','QT','Weight'],
    'Vif':[vif_age,vif_KM,vif_HP,vif_CC,vif_DR,vif_GR,vif_QT,vif_WT]}
Vif_df=pd.DataFrame(d1)
Vif_df


# None variable has VIF>20, No Collinearity, so consider all varaibles in Regression equation

# In[22]:


# 2) Residual Analysis
# Test for Normality of Residuals (Q-Q Plot) using residual model (model.resid)
sm.qqplot(model.resid,line='q') # 'q' - A line is fit through the quartiles # line = '45'- to draw the 45-degree diagonal line
plt.title("Normal Q-Q plot of residuals")
plt.show()


# In[23]:


list(np.where(model.resid>6000))  # outliar detection from above QQ plot of residuals


# In[24]:


list(np.where(model.resid<-6000))


# In[25]:


# Test for errors or Residuals Vs Regressors or independent 'x' variables or predictors 
# using Residual Regression Plots code graphics.plot_regress_exog(model,'x',fig)    # exog = x-variable & endog = y-variable


# In[26]:


fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'Age',fig=fig)
plt.show()


# In[27]:


fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'KM',fig=fig)
plt.show()


# In[28]:


fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'HP',fig=fig)
plt.show()


# In[29]:


fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'CC',fig=fig)
plt.show()


# In[30]:


fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'Doors',fig=fig)
plt.show()


# In[31]:


fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'Gears',fig=fig)
plt.show()


# In[32]:


fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'QT',fig=fig)
plt.show()


# In[33]:


fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'Weight',fig=fig)
plt.show()


# # Model Deletion Diagnostics (checking Outliers or Influencers)
# Two Techniques : 1. Cook's Distance & 2. Leverage value

# In[34]:


# 1. Cook's Distance: If Cook's distance > 1, then it's an outlier
# Get influencers using cook's distance
(c,_)=model.get_influence().cooks_distance
c


# In[35]:


# Plot the influencers using the stem plot
fig=plt.figure(figsize=(20,7))
plt.stem(np.arange(len(toyota4)),np.round(c,3))
plt.xlabel('Row Index')
plt.ylabel('Cooks Distance')
plt.show()


# In[36]:


# Index and value of influencer where C>0.5
np.argmax(c) , np.max(c)


# In[37]:


# 2. Leverage Value using High Influence Points : Points beyond Leverage_cutoff value are influencers
fig,ax=plt.subplots(figsize=(20,20))
fig=influence_plot(model,ax = ax)


# In[38]:


# Leverage Cuttoff Value = 3*(k+1)/n ; k = no.of features/columns & n = no. of datapoints
k=toyota4.shape[1]
n=toyota4.shape[0]
leverage_cutoff = (3*(k+1))/n
leverage_cutoff


# From the above plot, it is evident that points beyond leverage cutoff value=0.020905 are the outliers

# In[39]:


toyota4[toyota4.index.isin([80])]


# # Improving the Model

# In[40]:


# Creating a copy of data so that original dataset is not affected
toyota_new=toyota4.copy()
toyota_new


# In[41]:


# Discard the data points which are influencers and reassign the row number (reset_index(drop=True))
toyota5=toyota_new.drop(toyota_new.index[[80]],axis=0).reset_index(drop=True)
toyota5


# # Model Deletion Diagnostics and Final Model

# In[42]:


while model.rsquared < 0.90:
    for c in [np.max(c)>0.5]:
        model=smf.ols('Price~Age+KM+HP+CC+Doors+Gears+QT+Weight',data=toyota5).fit()
        (c,_)=model.get_influence().cooks_distance
        c
        np.argmax(c) , np.max(c)
        toyota5=toyota5.drop(toyota5.index[[np.argmax(c)]],axis=0).reset_index(drop=True)
        toyota5
    else:
        final_model=smf.ols('Price~Age+KM+HP+CC+Doors+Gears+QT+Weight',data=toyota5).fit()
        final_model.rsquared , final_model.aic
        print("Thus model accuracy is improved to",final_model.rsquared)


# In[43]:


final_model.rsquared # Model Accuracy is increased to 90.02%


# # Model Predictions

# In[44]:


# say New data for prediction is
new_data=pd.DataFrame({'Age':12,"KM":40000,"HP":80,"CC":1300,"Doors":4,"Gears":5,"QT":69,"Weight":1012},index=[0])
new_data


# In[45]:


# Manual Prediction of Price
final_model.predict(new_data)


# In[46]:


# Automatic Prediction of Price with 90.02% accurcy
pred_y=final_model.predict(toyota5)
pred_y

