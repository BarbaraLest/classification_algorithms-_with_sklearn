#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


# In[2]:


df = pd.read_csv('./iris.csv')
df.head()


# In[3]:


df.replace({'Iris-setosa': '1', 'Iris-versicolor': '2',  'ris-virginica': '3'}, inplace = True, regex = True)
df.head()


# In[4]:


x = df.values[:, 1:3]
y = df.values[:, 3]


# In[5]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 100)


# In[6]:


clf = KNeighborsClassifier()


# In[7]:


clf.fit(X_train, y_train)


# In[8]:


yprediction = clf.predict(X_test)


# In[9]:


print('Resultado previsto:\n', yprediction, '\n')
print('Resultado atual:\n', y_test, '\n')
print('Precisão do modelo de árvore de decisão ID3 para estes dados: ', accuracy_score(y_test, yprediction)*100)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




