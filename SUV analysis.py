#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib as plt
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


dataset = pd.read_csv('SUV.csv')


# In[5]:


dataset.head()


# In[6]:


x=dataset.iloc[:,[2,3]].values    
y=dataset.iloc[:,4].values 


# In[8]:


from sklearn.model_selection import train_test_split


# In[10]:


X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)


# In[12]:


from sklearn.preprocessing import StandardScaler


# In[13]:


sc = StandardScaler()


# In[15]:


X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)


# In[16]:


from sklearn.linear_model import LogisticRegression


# In[17]:


classifier = LogisticRegression()
classifier.fit(X_train,y_train)


# In[18]:


prediction = classifier.predict(X_test)


# In[19]:


from sklearn.metrics import accuracy_score


# In[20]:


accuracy_score(y_test,prediction)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




