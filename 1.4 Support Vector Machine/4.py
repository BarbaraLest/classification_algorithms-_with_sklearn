#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm


# In[22]:


df = pd.read_csv('./segment-challenge.csv')
df.head()


# In[23]:


df.replace({'cement': '1', 'path': '2',  'grass': '3',  'window': '4', 'foliage':'5','brickface':'6', 'sky': '7'}, inplace = True, regex = True)


# In[24]:


df.head()


# In[25]:


x = df.values[:, 1:20]
y = df.values[:, 20]


# In[26]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 100)


# In[27]:


clf = svm.SVC(kernel='sigmoid')


# In[28]:


clf.fit(X_train, y_train)


# In[29]:


yprediction = clf.predict(X_test)


# In[30]:


print('Resultado previsto:\n', yprediction, '\n')
print('Resultado atual:\n', y_test, '\n')
print('Precisão do modelo de árvore de decisão ID3 para estes dados: ', accuracy_score(y_test, yprediction)*100)


# In[ ]:





# In[ ]:





# In[ ]:




