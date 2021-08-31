#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


# In[2]:


df = pd.read_csv('./glass.csv')
df.head()


# In[3]:


df["'Type'"].unique()


# In[4]:


df.replace({'build wind float': '1', 'vehic wind float': '2',  'tableware': '3', 'build wind non-float': '4', 'headlampst': '5', 'containers': '6', }, inplace = True, regex = True)
df.head()


# In[5]:


x = df.values[:, 1:10]
y = df.values[:, 10]


# In[6]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 100)


# In[7]:


clf = KNeighborsClassifier(p=3)


# In[8]:


clf.fit(X_train, y_train)


# In[9]:


yprediction = clf.predict(X_test)


# In[10]:


print('Resultado previsto:\n', yprediction, '\n')
print('Resultado atual:\n', y_test, '\n')
print('Precisão do modelo de árvore de decisão ID3 para estes dados: ', accuracy_score(y_test, yprediction)*100)


# In[ ]:





# In[ ]:





# In[ ]:




