#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath('./introduction.ipynb'))))


# In[2]:


import numpy as np
import matplotlib.pyplot as plt


# In[3]:


np.random.seed(2022)

n = 5000

e = np.random.normal(size=(n,)) / 5
x = np.random.uniform(low=0.0, high=10.0, size=(n,))
z = np.random.uniform(low=0.0, high=10.0, size=(n,))

t = np.sqrt(2 * z + x * z + x * x + x) + e


# In[4]:


y = t*t / 10 +  x * x / 50 + e

# The endogeneity problem is clear, the latent error enters both treatment and outcome equally
plt.scatter(t,z, label ='raw data')
t_range = np.arange(-2,12)
y_range2 = t_range * t_range/10 + 0.08
y_range5 = t_range * t_range/10 + 0.5
y_range8 = t_range * t_range/10  + 1.3
plt.plot(t_range, y_range2, 'r--', label = 'truth, x=2')
plt.plot(t_range, y_range5, 'g--', label = 'truth, x=5')
plt.plot(t_range, y_range8, 'y--', label = 'truth, x=8')
plt.xlabel("t")
plt.ylabel("y")
plt.legend()
plt.show()


# In[5]:


import pandas as pd
data_dict = {
    'z': z,
    'w': x,
    'x': t,
    'y': y
}
data = pd.DataFrame(data_dict)
data.head()


# In[6]:


from ylearn.estimator_model.iv import NP2SLS


# In[7]:


iv = NP2SLS()
iv.fit(
    data=data,
    outcome='y',
    treatment='x',
    instrument='z',
    covariate='w',
    covar_basis=('Poly', 2),
    treatment_basis=('Poly', 2),
    instrument_basis=('Poly', 1),    
)


# In[8]:


n_test = 500
for i, x in enumerate([2, 5, 8]):
    t = np.linspace(0,10,num = 100)
    # y_true = t*t / 10 - x*t/10
    y_true = t * t / 10 + x * x / 50

    test_data = pd.DataFrame(
        {'x': t,
         'w': np.full_like(t, x),}
    )
    y_pred = iv.estimate(data=test_data, quantity='CF')
    plt.plot(t, y_true, label='true y, x={0}'.format(x),color='C'+str(i))
    plt.plot(t, y_pred, label='pred y, x={0}'.format(x),color='C'+str(i),ls='--')
plt.xlabel('t')
plt.ylabel('y')
plt.legend()
plt.show()


# Test for the special case: 2SLS

# In[9]:


n = 5000

# Initialize exogenous variables; normal errors, uniformly distributed covariates and instruments
e = np.random.normal(size=(n,))
x = np.random.uniform(low=0.0, high=10.0, size=(n,)) + e
z = np.random.uniform(low=0.0, high=10.0, size=(n,))

# Initialize treatment variable
# t = np.sqrt((x + 2) * z) + e
t = x * z + e + z
y = t * x / 2 +  x / 5 + e

data_dict = {
    'z': z,
    'w': x,
    'x': t,
    'y': y
}
data = pd.DataFrame(data_dict)
data.head()


# In[10]:


iv1 = NP2SLS()
iv1.fit(
    data=data,
    outcome='y',
    treatment='x',
    instrument='z',
    covariate='w',
    covar_basis=('Poly', 1),
    treatment_basis=('Poly', 1),
    instrument_basis=('Poly', 1),    
)


# In[11]:


n_test = 500
for i, x in enumerate([2, 5, 8]):
    t = np.linspace(0,10,num = 100)
    # y_true = t*t / 10 - x*t/10
    # y_true = t * t / 10 + x * x / 50
    y_true = t * x / 2 +  x / 5

    
    test_data = pd.DataFrame(
        {'x': t,
         'w': np.full_like(t, x),}
    )
    y_pred = iv1.estimate(data=test_data, quantity='CF')
    plt.plot(t, y_true, label='true y, x={0}'.format(x),color='C'+str(i))
    plt.plot(t, y_pred, label='pred y, x={0}'.format(x),color='C'+str(i),ls='--')
plt.xlabel('t')
plt.ylabel('y')
plt.legend()
plt.show()


# ### We compare the IV results in last section to results of naive LinearRegression to indicate the effetiveness of IV.

# In[12]:


from sklearn.linear_model import LinearRegression


# In[13]:


n = 5000

# Initialize exogenous variables; normal errors, uniformly distributed covariates and instruments
e = np.random.normal(size=(n,))
# x = np.random.uniform(low=0.0, high=10.0, size=(n,))
z = np.random.uniform(low=0.0, high=10.0, size=(n,))

# Initialize treatment variable
# t = np.sqrt((x + 2) * z) + e


# In[14]:


x = np.ones((n, )) * 2 + e
t = (x * z + e + z)
y = t * x / 2 +  x / 5 + e

l2 = LinearRegression()
l2.fit(t.reshape(-1, 1), y.squeeze())


# In[15]:


x = np.ones((n, )) * 5 + e
t = (x * z + e + z)
y = (t * x / 2 +  x / 5 + e)

l5 = LinearRegression()
l5.fit(t.reshape(-1, 1), y.squeeze())


# In[16]:


x = np.ones((n, )) * 8 + e
t = (x * z + e + z)
y = t * x / 2 +  x / 5 + e

l8 = LinearRegression()
l8.fit(t.reshape(-1, 1), y.squeeze())


# In[17]:


n_test = 500
model = [l2, l5, l8]
for i, x in enumerate([2, 5, 8]):
    t = np.linspace(0,10,num = 100)
    x_ = np.full_like(t, x)
    y_true = t * x_ / 2 +  x_ / 5

    y_pred = model[i].predict(t.reshape(-1, 1))
    plt.plot(t, y_true, label='true y, x={0}'.format(x),color='C'+str(i))
    plt.plot(t, y_pred, label='pred y, x={0}'.format(x),color='C'+str(i),ls='--')
plt.xlabel('t')
plt.ylabel('y')
plt.legend()
plt.show()

