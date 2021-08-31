#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB


# In[11]:


df = pd.read_csv('./ionosphere.csv')
df.head()


# In[12]:


df['class'].unique()


# In[13]:


df['class'].replace({'g': '0', 'b': '1'}, inplace = True, regex = True)
df.dropna(subset = ["class"], inplace=True)
df.head()


# In[14]:


x = df.values[:, 1:35]
y = df.values[:, 35]


# In[15]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 100)


# In[16]:


clf = GaussianNB()


# In[17]:


clf.fit(X_train, y_train)
yprediction = clf.predict(X_test)


# In[19]:


print('Resultado previsto:\n', yprediction, '\n')
print('Resultado atual:\n', y_test, '\n')
print('Precisão do modelo de árvore de decisão ID3 para estes dados: ', accuracy_score(y_test, yprediction)*100)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




