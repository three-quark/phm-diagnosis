#!/usr/bin/env python
# coding: utf-8

# # YLearn Case Study - Infant Health and Development Program

# ## Content
#   
#  1. Summary  
#  2. Dataset  
#  3. Analysis  
#      A. No Treatment  
#      B. AB Test Analysis  
#      C. Best choice  
#      D. Policy choice  
#  4. Comparison

# ## 1. Summary
# 
# This notebook utilizes IHDP dataset to demonstrate the usage on Policy Interpreter in Precision in Estimation of Heterogenous Effects.

# ## 2. Dataset
# 
# Hill introduced a semi-synthetic dataset constructed from the Infant Health and Development Program (IHDP). This dataset is based on a randomized experiment investigating the effect of home visits by specialists on future cognitive scores. The IHDP simulation is considered the de-facto standard benchmark for neural network treatment effect estimation methods.
# 
# Depend on the fields of y_factual and y_cfactual in this dataset, we will look into the usage of the Policy interpreter and compare different policy.

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import numpy as np
import pandas as pd
from matplotlib import pyplot  as plt

from sklearn.model_selection import train_test_split
from ylearn import Why


# In[2]:


# load all ihadp data
df = pd.DataFrame()
for i in range(1, 10):
    data = pd.read_csv(r'data/ihdp_npci_' + str(i) + '.csv', header=None)
    df = pd.concat([data, df])
df = df.head(10000)
cols =  ["treatment", "y_factual", "y_cfactual", "mu0", "mu1"] + ['x' + str(i) for i in range(25)]
df.columns = cols
print(df.shape)


# In[3]:


df.head()


# ### 3. Analysis

# In[4]:


import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import numpy as np
import pandas as pd
from matplotlib import pyplot  as plt

from sklearn.model_selection import train_test_split
from ylearn import Why


# In[5]:


# load all ihadp data
df = pd.DataFrame()
for i in range(1, 10):
    data = pd.read_csv(r'data/ihdp_npci_' + str(i) + '.csv', header=None)
    df = pd.concat([data, df])
df = df.head(10000)
cols =  ["treatment", "y_factual", "y_cfactual", "mu0", "mu1"] + ['x' + str(i) for i in range(25)]
df.columns = cols
print(df.shape)


# In[6]:


df.head()


# #### A. No Treament
# First, look at mean value of y_factual where unit didn't get treament.

# In[7]:


df_cp= df.copy()
def policy_no_t(row):
    return row['y_factual'] if row['treatment'] == 0 else row['y_cfactual']
v_no_t = df_cp.apply(policy_no_t, axis = 1)
v_no_t.mean()


# #### B. AB Test Analysis
# 
# Regard the data as A/B test and compare the mean value with and without treatment.

# In[8]:


df_full = df
(df_full[df_full['treatment'] ==1]['y_factual'].mean(),
 df_full[df_full['treatment'] ==0]['y_factual'].mean(),
 df_full[df_full['treatment'] ==1]['y_factual'].mean()-df_full[df_full['treatment'] ==0]['y_factual'].mean())


# Conclusion: in the full dataset, the effect of participating in treatment is significantly better than that of not participating in treatment.   
# Then apply the treatment strategy to all unit.

# In[9]:


def policy_from_AB(row):
    return row['y_factual'] if row['treatment'] == 1 else row['y_cfactual']
v_AB = df_full.apply(policy_from_AB, axis = 1)
v_AB.mean()


# B. AB Test Analysis
# Regard the data as A/B test and compare the mean value with and without treatment.

# In[10]:


train_data = df_full.drop(['y_cfactual','mu0','mu1'],axis=1).reset_index(drop=True)


# #### C. Best line
# According to the actual value and factual value, the best treatment choice is drawn up. That is, choose the maximum of the two as the best performance reference.

# In[11]:


def policy_best(row):
    return row['y_factual'] if row['y_factual'] > row['y_cfactual'] else row['y_cfactual']
v_best = df_cp.apply(policy_best, axis = 1)
v_best.mean()


# #### D. Policy interpreter 
# Through why's policy interpreter, formulate heterogeneous strategies.

# In[12]:


why=Why()
why.fit(train_data,outcome='y_factual',treatment='treatment')

pi=why.policy_interpreter(train_data, max_depth=2)
plt.figure(figsize=(30, 20), )
pi.plot()


# In[13]:


def policy_from_ti(row):
    if row['x5'] <= 2.372:
        if row['x0'] > 1.5:
            return 0
    else:
        if row['x1'] > 0.197:
            return 0
    return 1
t_from_policy = df.apply(policy_from_ti, axis = 1)
df_cp= df.copy()
df_cp['policy'] = t_from_policy
v_policy = df_cp.apply(lambda d: d['y_factual'] if d['treatment']==d['policy'] 
               else d['y_cfactual'], 
               axis=1)
v_policy.mean()


# ## 4. comparison

# In[14]:


summary=pd.Series({
    'No Treament': v_no_t.mean(),
    'Policy from AB Test': v_AB.mean(),
    'Policy from Interpreter': v_policy.mean(),
    'Best Line': v_best.mean()
    },name=f'Mean of Y').to_frame()
summary.plot(kind='bar',legend=False)
summary


# Conclusion: through strategy learning and treatment, the overall score of the unit can be improved slightly.

# In[15]:


pi=why.policy_interpreter(train_data, max_depth=3)
plt.figure(figsize=(30, 20), )
pi.plot()


# In[16]:


def policy_from_ti3(row):
    if row['x5'] <= 2.372:
        if row['x0'] > 1.5:
            return 0
    else:
        if row['x1'] > 0.197:
            if row['x4']> -0.692:
                return 0
    return 1
t_from_policy3 = df.apply(policy_from_ti3, axis = 1)
df_cp3= df.copy()
df_cp3['policy'] = t_from_policy3
v_policy3 = df_cp3.apply(lambda d: d['y_factual'] if d['treatment']==d['policy'] 
               else d['y_cfactual'], 
               axis=1)
v_policy3.mean()


# In[17]:


summary=pd.Series({
    'No Treament': v_no_t.mean(),
    'Policy from AB Test': v_AB.mean(),
    'Policy from Interpreter[md2]': v_policy.mean(),
    'Policy from Interpreter[md3]': v_policy3.mean(),
    'Best Line': v_best.mean()
    },name=f'Mean of Y').to_frame()
summary.plot(kind='bar',legend=False)
summary


# Conclusion: increasing the maximum depth of the policy tree, may improve the performance.

# ## compare dml estimator

# In[18]:


why_dml=Why(estimator='dml')
why_dml.fit(train_data,outcome='y_factual',treatment='treatment')
pi_dml=why_dml.policy_interpreter(train_data, max_depth=2)
plt.figure(figsize=(30, 20), )
pi_dml.plot()


# In[19]:


def policy_from_ti_dml(row):
    if row['x0'] <= 1.5:
        if row['x5'] > 2.537:
            return 0
    else:
        if row['x1'] > -0.796:
            return 0
    return 1
t_from_policy_dml = df.apply(policy_from_ti_dml, axis = 1)
df_cp= df.copy()
df_cp['policy'] = t_from_policy_dml
v_policy_dml = df_cp.apply(lambda d: d['y_factual'] if d['treatment']==d['policy'] 
               else d['y_cfactual'], 
               axis=1)
v_policy_dml.mean()


# In[ ]:


why_dr=Why(estimator='dr')
why_dr.fit(train_data,outcome='y_factual',treatment='treatment')
pi_dr=why_dr.policy_interpreter(train_data, max_depth=2)
plt.figure(figsize=(10, 6), )
pi_dr.plot()


# In[21]:


def policy_from_ti_dr(row):
    if row['x0'] > 1.5:
        if row['x1'] > 1.196:
            return 0
    return 1
t_from_policy_dr = df.apply(policy_from_ti_dr, axis = 1)
df_cp= df.copy()
df_cp['policy'] = t_from_policy_dr
v_policy_dr = df_cp.apply(lambda d: d['y_factual'] if d['treatment']==d['policy'] 
               else d['y_cfactual'], 
               axis=1)
v_policy_dr.mean()


# In[22]:


why_meta_learner=Why(estimator='meta_leaner')
why_meta_learner.fit(train_data,outcome='y_factual',treatment='treatment')
pi_meta_learner=why_meta_learner.policy_interpreter(train_data, max_depth=2)
plt.figure(figsize=(30, 20), )
pi_meta_learner.plot()


# In[23]:


def policy_from_ti_meta(row):
    if row['x5'] > 2.372:
        if row['x0'] > 1.5:
            return 0
    else:
        if row['x1'] > 0.197:
            return 0        
    return 1
t_from_policy_meta = df.apply(policy_from_ti_meta, axis = 1)
df_cp= df.copy()
df_cp['policy'] = t_from_policy_meta
v_policy_meta = df_cp.apply(lambda d: d['y_factual'] if d['treatment']==d['policy'] 
               else d['y_cfactual'], 
               axis=1)
v_policy_meta.mean()


# In[25]:


summary=pd.Series({
    'No Treament': v_no_t.mean(),
    'Policy from AB Test': v_AB.mean(),
    'Policy from Interpreter[md2]': v_policy.mean(),
    'Policy from Interpreter[md3]': v_policy3.mean(),
    'Policy from Interpreter[dml]': v_policy_dml.mean(),    
    'Policy from Interpreter[dr]': v_policy_dr.mean(),
    'Policy from Interpreter[meta]': v_policy_meta.mean(),
    'Policy Best': v_best.mean()
    },name=f'Mean of Y').to_frame()
summary.plot(kind='bar',legend=False)
summary


# In[ ]:




