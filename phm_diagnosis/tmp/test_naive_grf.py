#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath('./test_score.ipynb'))))


# In[2]:


import numpy as np
import pandas as pd

from ylearn.estimator_model._naive_forest import _GrfTree, NaiveGrf
from ylearn.estimator_model import CausalTree
from ylearn.exp_dataset.exp_data import sq_data
from ylearn.utils._common import to_df
from ylearn.estimator_model._naive_forest.utils import grad, grad_coef, inverse_grad

from numpy.linalg import lstsq, inv


# In[3]:


from copy import deepcopy


n = 2000
d = 10     
n_x = 1
y, x, v = sq_data(n, d, n_x)

true_te = lambda X: np.hstack([X[:, [0]]**2 + 1, np.ones((X.shape[0], n_x - 1))])
w = deepcopy(v)
v_test = v[:min(100, n)].copy()
v_test[:, 0] = np.linspace(np.percentile(v[:, 0], 1), np.percentile(v[:, 0], 99), min(100, n))


# In[4]:


data = to_df(treatment=x, outcome=y, v=v)
test_data = to_df(v=v_test)

outcome = 'outcome'
treatment = 'treatment'
adjustment = data.columns[2:]


# In[5]:


from sklearn.preprocessing import OneHotEncoder
# oh = OneHotEncoder()
# x = oh.fit_transform(x).toarray()


# In[6]:


gt = _GrfTree()
gt._fit_with_array(x=x, y=y.squeeze(), w=w, v=v, i=1)


# In[7]:


grf = NaiveGrf(n_jobs=1, n_estimators=100)
grf.verbose = 1


# In[8]:


grf.fit(
    data=data, outcome=outcome, treatment=treatment, adjustment=adjustment, covariate=adjustment
)


# In[9]:


effect = grf._prepare4est(test_data)


# In[10]:


import matplotlib.pyplot as plt

true_te = lambda X: np.hstack([X[:, [0]]**2 + 1, np.ones((X.shape[0], n_x - 1))])

for t in range(n_x):
    plt.plot(v_test[:, 0], effect[:, t])
    plt.plot(v_test[:, 0], true_te(v_test)[:, t])
plt.show()

