#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron


# In[2]:


df = pd.read_csv('./cpu-vendedor.csv')
df.head()


# In[3]:


df['vendor'].unique()


# In[4]:


df = df.drop(['vendor'], axis=1)
df.head()


# In[5]:


x = df.values[:, 1:7]
y = df.values[:, 7]


# In[16]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 100)


# In[17]:


clf = Perceptron(tol=1e-3, random_state=0)


# In[18]:


clf.fit(X_train, y_train)


# In[19]:


yprediction = clf.predict(X_test)


# In[20]:


print('Resultado previsto:\n', yprediction, '\n')
print('Resultado atual:\n', y_test, '\n')
print('Precisão do modelo de árvore de decisão ID3 para estes dados: ', accuracy_score(y_test, yprediction)*100)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




