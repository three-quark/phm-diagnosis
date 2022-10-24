#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath('./test_score.ipynb'))))


# In[2]:


import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from ylearn.estimator_model import RLoss
from ylearn.estimator_model import meta_learner, double_ml, doubly_robust, causal_tree
from ylearn.exp_dataset.exp_data import single_binary_treatment


# ## Single binary treatment

# In[3]:


train1, val1, treatment_effect1 = single_binary_treatment()
def exp_te(x): return np.exp(2*x[0])
n = 1000
n_x = 4
X_test1 = np.random.uniform(0, 1, size=(n, n_x))
X_test1[:, 0] = np.linspace(0, 1, n)
data_test_dict = {
    'c_0': X_test1[:, 0],
    'c_1': X_test1[:, 1],
    'c_2': X_test1[:, 2],
    'c_3': X_test1[:, 3],
}
data_test1 = pd.DataFrame(data_test_dict)
true_te = np.array([exp_te(x_i) for x_i in X_test1])


# In[4]:


adjustment = train1.columns[:-7]
covariate = train1.columns[-7:-3]
# t_effect1 = train1['t_effect']
treatment = 'treatment'
outcome = 'outcome'
train1.head()


# In[5]:


rloss = RLoss(
    x_model=RandomForestClassifier(),
    y_model=RandomForestRegressor(),
    cf_fold=1,
    is_discrete_treatment=True
)
rloss.fit(
    data=val1,
    outcome=outcome,
    treatment=treatment,
    adjustment=adjustment,
    covariate=covariate,
)


# In[6]:


dml = double_ml.DoubleML(
    x_model=RandomForestClassifier(),
    y_model=RandomForestRegressor(),
    cf_fold=1,
    is_discrete_treatment=True
)
slearner = meta_learner.SLearner(
    model=RandomForestRegressor(),
)
tlearner = meta_learner.TLearner(
    model=RandomForestRegressor()
)
xlearner = meta_learner.XLearner(
    model=RandomForestRegressor()
)
dr = doubly_robust.DoublyRobust(
    x_model=RandomForestClassifier(),
    y_model=RandomForestRegressor(),
    yx_model=RandomForestRegressor(),
)
ct = causal_tree.CausalTree()
models = [dml, slearner, tlearner, xlearner, ct, dr]


# In[7]:


for model in models:
    model.fit(
    data=train1,
    treatment=treatment,
    outcome=outcome,
    adjustment=adjustment,
    covariate=covariate
)


# In[8]:


for model in models:
    print(f'The score of {model.__repr__()} is {rloss.score(model)}')

