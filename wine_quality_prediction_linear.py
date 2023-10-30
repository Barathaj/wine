#!/usr/bin/env python
# coding: utf-8

# In[28]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


data=pd.read_csv("C:\\BARATHRAJ\\DATASET\\WineQT.csv")
data


# In[5]:


data.isnull().sum()


# In[7]:


data.describe()


# In[15]:


data.hist(bins=20, figsize=(10, 10))
plt.show()


# In[57]:


plt.bar(data['quality'], data['alcohol'])
plt.show()


# In[58]:


x = data.drop(columns=['quality'],axis=1)
y= data['quality']
y


# In[55]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
x_train


# In[59]:


linear=LinearRegression()
linear.fit(x_train,y_train)
y_pred=linear.predict(x_test)
r2=metrics.r2_score(y_test,y_pred)
print(r2*100)


# In[73]:


from sklearn.ensemble import RandomForestRegressor


forest=RandomForestRegressor()
forest.fit(x,y)
y_pred=forest.predict(x_test)
metrics.r2_score(y_test,y_pred)


# In[78]:


import matplotlib.pyplot as plt

# Assuming y_true contains actual wine quality values and y_pred contains predicted values
# y_true = ...
# y_pred = ...

plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred, alpha=0.5, color='b')
plt.xlabel('Actual Wine Quality')
plt.ylabel('Predicted Wine Quality')
plt.title('Wine Quality Prediction Scatter Plot')
plt.show()

