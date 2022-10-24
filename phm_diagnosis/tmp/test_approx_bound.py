#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath('./introduction.ipynb'))))


# In[2]:


import numpy as np

from ylearn.estimator_model.approximation_bound import ApproxBound
from ylearn.exp_dataset.exp_data import meaningless_discrete_dataset_


# In[3]:


num = 1000
np.random.seed(2022)


# In[4]:


data = meaningless_discrete_dataset_(num=num,
                                    confounder_n=3,
                                    treatment_effct=[2, 5, -8],
                                    random_seed=0)
data.head()


# In[5]:


treatment = 'treatment'
w = ['w_0', 'w_1', 'w_2']
outcome = 'outcome'
data[treatment].value_counts()


# In[6]:


from sklearn.linear_model import LinearRegression
y_model = LinearRegression()
bound = ApproxBound(y_model=y_model)


# In[7]:


bound.fit(
    data=data,
    treatment=treatment,
    outcome=outcome,
    # covariate=w,
)


# In[8]:


np.unique(np.array([4, 3, 4]), return_counts=True)[1] / 5


# In[9]:


bound.estimate()


# In[10]:


from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

bound1 = ApproxBound(
    y_model=RandomForestRegressor(),
    x_model=RandomForestClassifier()
)
bound1.fit(
    data=data,
    treatment=treatment,
    outcome=outcome,
    covariate=w,
)


# In[11]:


b_l, b_u = bound1.estimate()


# In[12]:


b_l.mean()


# In[13]:


b_u.mean()

