#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing the file
import pandas as pd
data=pd.read_csv("nse_data.csv",header=0)


# In[2]:


#importing the modules
import seaborn as sns
import matplotlib.pyplot as plt

#correltaion of the data
#correlations=data.corr()
#sns.heatmap(correlations)
#plt.yticks(rotation=0)
#plt.xticks(rotation=90)
#plt.show()


# In[3]:


#arranging the data in ascendng order so that the 2016 data comes first followed by the 2017 data
data.index=data['TIMESTAMP']
data=data.sort_index(ascending=True,axis=0)
data.head()


# In[6]:


#storing the column "TIMESTAMP" into other variable
year=data["TIMESTAMP"]


# In[7]:


for j ,i in zip(year,range(len(data))):
    if j[0:4]=="2017":
        global ind
        ind=i
        break
        
#extracting only the 2016 data      
X=data.iloc[0:ind,0:12].drop("TIMESTAMP",axis=1).drop("SYMBOL",axis=1).drop("SERIES",axis=1)


# In[11]:


X=(X-X.mean()/X.std())


# In[12]:



Q1 = X.quantile(0.25)
Q3 = X.quantile(0.75)
IQR = Q3 - Q1
X = X[~((X< (Q1 - 1.5 * IQR)) |(X> (Q3 + 1.5 * IQR))).any(axis=1)].dropna()


# In[18]:


#extracting the 2017 data
Y_orig=data.iloc[ind:ind*2,0:12].drop("TIMESTAMP",axis=1).drop("SYMBOL",axis=1).drop("SERIES",axis=1)


Y=Y_orig["CLOSE"].dropna()


# In[15]:


Y=(Y-Y.mean())/Y.std()


# In[19]:



Q1 = Y.quantile(0.25)
Q3 = Y.quantile(0.75)
IQR = Q3 - Q1
Y = Y[~((Y< (Q1 - 1.5 * IQR)) |(Y> (Q3 + 1.5 * IQR)))].dropna()
Y=Y[0:len(X)]


# In[20]:


#importing the module to split the data
from sklearn.model_selection import train_test_split
#splitting the data into train and test  with train size as 40% and rest (60%) data will be tested
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.4,random_state=1)


# In[21]:


from sklearn.preprocessing import StandardScaler
model=StandardScaler()
X_train=model.fit_transform(X_train)
X_test=model.transform(X_test)


# In[22]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

classifier=Sequential()
classifier.add(Dense(units=128,kernel_initializer='uniform',activation='relu',input_dim=X.shape[1]))
classifier.add(Dense(units=128,kernel_initializer='uniform',activation='relu'))
classifier.add(Dense(units=128,kernel_initializer='uniform',activation='relu'))
classifier.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))


# In[23]:


classifier.compile(optimizer='adam',loss='mean_absolute_errr',metrics=['accuracy'])
classifier.fit(X_train,Y_train,batch_size=10,epochs=1)


# In[24]:


import numpy as np
Y_pred=classifier.predict(X_test)
Y_orig['Y_pred']=0
Y_orig.iloc[(len(Y_orig)-len(Y_pred)):,-1:]=Y_pred
prediction=pd.DataFrame(Y_orig.dropna())


# In[26]:



plt.plot(prediction["Y_pred"],label="Prediction")
plt.legend(loc="best")


# In[27]:


plt.figure(figsize=[10,10])
plt.plot(X["HIGH"],label="HIGH_2016")
plt.plot(prediction["HIGH"],label="HIGH_2017_prediction")
plt.xlabel("TIME SERIES_2017")
plt.title("Statistical arbitrage opportunities")
plt.legend(loc='upper left')
plt.show()


# In[28]:


plt.figure(figsize=[10,10])
plt.plot(X["TOTALTRADES"],label="2016_total_trades")
plt.plot(prediction["TOTALTRADES"],label="2017_total_trades")
plt.xlabel("TIME SERIES_2017")
plt.title("Statistical arbitrage opportunities")
plt.legend(loc='upper left')
plt.show()


# In[ ]:
