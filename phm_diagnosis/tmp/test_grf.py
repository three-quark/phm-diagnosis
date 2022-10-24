#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath('./test_score.ipynb'))))

import numpy as np
import pandas as pd

from ylearn.estimator_model._generalized_forest.tree._grf_tree import GrfTree
from ylearn.estimator_model._generalized_forest._grf import GRForest
from ylearn.exp_dataset.exp_data import sq_data
from ylearn.utils._common import to_df
from ylearn.estimator_model._naive_forest.utils import grad, grad_coef, inverse_grad

from numpy.linalg import lstsq, inv

from copy import deepcopy


n = 2000
d = 10     
n_x = 1
y, x, v = sq_data(n, d, n_x)

true_te = lambda X: np.hstack([X[:, [0]]**2 + 1, np.ones((X.shape[0], n_x - 1))])
w = deepcopy(v)
v_test = v[:min(100, n)].copy()
v_test[:, 0] = np.linspace(np.percentile(v[:, 0], 1), np.percentile(v[:, 0], 99), min(100, n))
data = to_df(treatment=x, outcome=y, v=v)
test_data = to_df(v=v_test)

outcome = 'outcome'
treatment = 'treatment'
adjustment = data.columns[2:]
from sklearn.preprocessing import OneHotEncoder
oh = OneHotEncoder()
# x = oh.fit_transform(x).toarray().astype(np.float32)



gtre = GrfTree(max_depth=100000, max_leaf_nodes=np.int(100000))
gtre._fit_with_array(x, y, w=None, v=v)
gtre.tree_.predict(v[:50].astype(np.float32)).squeeze()


# In[2]:


grf = GRForest(
    n_jobs=1, 
    honest_subsample_num=None,
    min_samples_split=10, 
    sub_sample_num=0.5, 
    n_estimators=100, 
    random_state=2022, 
    min_impurity_decrease=1e-10, 
    max_depth=100, 
    max_leaf_nodes=100, 
    verbose=0,
)
grf.fit(
    data=data, outcome=outcome, treatment=treatment, adjustment=adjustment, covariate=adjustment
)


# In[3]:


e = grf.estimators_[0]
pred = e._predict_with_array(None, v_test).reshape(-1, 1)
y_pred = e.leaf_record.reshape(1, -1)


# In[4]:


temp = y_pred == pred
z = temp / np.count_nonzero(temp, axis=1).reshape(-1, 1)


# In[5]:


temp.shape


# In[6]:


alpha = np.zeros((v_test.shape[0], 2000))
idx = grf.sub_sample_idx[0]
alpha[:, idx].shape


# In[7]:


effect = grf._prepare4est(test_data)


# In[8]:


import matplotlib.pyplot as plt

true_te = lambda X: np.hstack([X[:, [0]]**2 + 1, np.ones((X.shape[0], n_x - 1))])

for t in range(n_x):
    plt.plot(v_test[:, 0], effect[:, t])
    plt.plot(v_test[:, 0], true_te(v_test)[:, t])
plt.show()


# In[47]:


grf_honest = GRForest(
    n_jobs=1, 
    honest_subsample_num=0.35,
    min_samples_split=10, 
    sub_sample_num=0.75, 
    n_estimators=100, 
    random_state=2022, 
    min_impurity_decrease=1e-10, 
    max_depth=100, 
    max_leaf_nodes=100, 
    verbose=0,
)
grf_honest.fit(
    data=data, outcome=outcome, treatment=treatment, adjustment=adjustment, covariate=adjustment
)


# In[48]:


effect_honest = grf_honest._prepare4est(test_data)
for t in range(n_x):
    plt.plot(v_test[:, 0], effect_honest[:, t])
    plt.plot(v_test[:, 0], true_te(v_test)[:, t])
plt.show()

