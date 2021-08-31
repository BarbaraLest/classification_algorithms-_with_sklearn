#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import CategoricalNB


# In[8]:


df = pd.read_csv('./weather.csv')
df.head()


# In[9]:


df = df.drop(['outlook', 'windy', 'play'], axis=1)
df.head()


# In[10]:


x = df.values[:, 1:2]
y = df.values[:, 2]


# In[11]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 100)


# In[12]:


clf = CategoricalNB()


# In[13]:


clf.fit(X_train, y_train)


# In[14]:


yprediction = clf.predict(X_test)


# In[16]:


print('Resultado previsto:\n', yprediction, '\n')
print('Resultado atual:\n', y_test, '\n')
print('Precisão do modelo de árvore de decisão ID3 para estes dados: ', accuracy_score(y_test, yprediction)*100)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




