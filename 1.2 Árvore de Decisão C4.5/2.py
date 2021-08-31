#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree


# In[3]:


df = pd.read_csv('./unbalanced.csv')
df.head()


# In[4]:


df.replace({'Active': '0', 'Inactive': '1'}, inplace = True, regex = True)


# In[5]:


df.head()


# In[6]:


x = df.values[:, 1:33]
y = df.values[:, 33]


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 100)


# In[8]:


clf_entropy = tree.DecisionTreeClassifier(criterion = 'entropy')


# In[9]:


clf_entropy.fit(X_train, y_train)


# In[10]:


yprediction = clf_entropy.predict(X_test)


# In[11]:


print('Resultado previsto:\n', yprediction, '\n')
print('Resultado atual:\n', y_test, '\n')
print('Precisão do modelo de árvore de decisão ID3 para estes dados: ', accuracy_score(y_test, yprediction)*100)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




