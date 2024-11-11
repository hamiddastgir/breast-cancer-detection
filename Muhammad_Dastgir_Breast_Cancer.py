#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from ucimlrepo import fetch_ucirepo 
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random


# ## Loading Dataset from UC Irvine Machine Learning Repository and Assigning Values

# In[5]:


breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)
X = breast_cancer_wisconsin_diagnostic.data.features 
y = breast_cancer_wisconsin_diagnostic.data.targets 


# In[6]:


X


# In[7]:


y


# ### Creating training and testing data

# In[8]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 69)


# In[9]:


training_model = LogisticRegression()
training_model.fit(X_train, y_train)


# In[10]:


predicted_y = training_model.predict(X_test)
predicted_y # for reference


# In[11]:


accuracy = training_model.score(X_test, y_test)
accuracy


# In[12]:


Cs = np.logspace(-4, 4, 10)


# In[13]:


l1_model = LogisticRegressionCV(
    Cs=Cs,
    cv=5,                  
    penalty='l1',        
    solver='liblinear',  
    max_iter=1000,
    scoring='accuracy', 
    refit=True
)


# In[14]:


l1_model.fit(X_train, y_train)


# In[15]:


l2_model = LogisticRegressionCV(
    Cs=Cs,
    cv=5,
    penalty='l2',
    solver='lbfgs',     
    max_iter=1000,
    scoring='accuracy',
    refit=True
)


# In[16]:


l2_model.fit(X_train, y_train)


# In[17]:


l2_model.fit(X_train, y_train)


# In[18]:


optimal_C_l1 = l1_model.C_[0]
optimal_C_l2 = l2_model.C_[0]

print(f'Optimal C for L1 penalty: {optimal_C_l1}')
print(f'Optimal C for L2 penalty: {optimal_C_l2}')


# In[19]:


y_pred_l1 = l1_model.predict(X_test)
y_pred_l2 = l2_model.predict(X_test)

accuracy_l1 = accuracy_score(y_test, y_pred_l1)
accuracy_l2 = accuracy_score(y_test, y_pred_l2)

print(f'L1 Penalty Accuracy: {accuracy_l1:.2f}')
print(f'L2 Penalty Accuracy: {accuracy_l2:.2f}')


# In[20]:


n_features_l1 = np.sum(l1_model.coef_ != 0)
n_features_l2 = np.sum(l2_model.coef_ != 0)

print(f'Number of features used in L1 model: {n_features_l1}')
print(f'Number of features used in L2 model: {n_features_l2}')


# ## Analysis

# The lasso regularized model was more accurate, and it had more non-zero coefficients. The lasso regularized model also benefitted from a higher lamda (inverse C - used above)

# In[21]:


y_pred_l1


# In[22]:


y_pred_l2

