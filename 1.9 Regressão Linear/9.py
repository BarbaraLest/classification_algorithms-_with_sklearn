#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer
from sklearn.linear_model import LinearRegression 


# In[2]:


df = pd.read_csv('./cpu.csv')
df.head()


# In[3]:


x = df.values[:, 1:7]
y = df.values[:, 7]


# In[4]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 100)


# In[5]:


clf = LinearRegression()


# In[6]:


clf.fit(X_train, y_train)


# In[15]:


y_predict = clf.predict(X_test)
def my_custom_loss_func(Y_test, y_predict):
     diff = np.abs(y_test - y_predict).max()
     return np.log1p(diff)
score = make_scorer(my_custom_loss_func, greater_is_better=False)
my_custom_loss_func(y_test, y_predict)


# In[19]:


print('Resultado previsto:\n', y_predict, '\n')
print('Resultado atual:\n', y_test, '\n')
print('Precisão do modelo de árvore de decisão ID3 para estes dados: ', my_custom_loss_func(y_test, y_predict))


# In[ ]:





# In[ ]:





# In[ ]:




