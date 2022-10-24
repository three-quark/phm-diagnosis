#!/usr/bin/env python
# coding: utf-8

# # YLearn Case Study - Marketing Promotion

# ## 1. Summary
# 
# ...

# ## 2. Dataset
# 
# ...

# In[1]:


import numpy as np
import pandas as pd
from matplotlib import pyplot  as plt
import warnings

from sklearn.model_selection import train_test_split

from ylearn import Why
from ylearn.uplift import plot_gain, auuc_score


# In[2]:


warnings.filterwarnings('ignore')

# pd.set_option('display.max.columns',100)


# In[3]:


data = pd.read_csv('data/marketing_promotion.csv.zip')
outcome = 'conversion'


# In[4]:


data.head(10)


# In[5]:


data.groupby(['offer', outcome,]).agg([np.size, ])['history'].unstack()


# In[6]:


train_data,test_data=train_test_split(data,test_size=0.5,random_state=123)

print('train_data:',train_data.shape)
print('test_data: ',test_data.shape)


# ## 3. Causal Effect analysis of BOGO and Discount

# ### 3.1 Learn Why
# Firstly, create an instance of `Why`. Then train the model with `fit()` method, which defines `treatment=’offer’.` The printed logs show the information of identified results and used eatimator. To visualize the causal relations, use `plot_causal_graph()` to obtain the causal graph.

# In[7]:


why=Why(estimator='slearner',estimator_options=dict(model='lightgbm'))
why.fit(train_data,outcome,treatment='offer')


# In[8]:


why.plot_causal_graph()


# ### 3.2 Estimate causal effect
# 
# There are four types of card categoties: Blue, Sliver, Gold and Platinum. Taking Blue as control variable and the rest three as treatment variables, method `causal_effect()` outputs three causal effect estimations.  From the results, we find that card upgrade will increase the personal transaction amount. Gold card has the strongest effect.

# In[9]:


effect = why.causal_effect(control='No Offer', return_detail=True)
effect = effect.loc['offer'].sort_values(by='mean')
details = effect.pop('detail')

plt.figure(figsize=(10, 5))
plt.violinplot(details.tolist(), showmeans=True)
plt.ylabel('Effect distribution')
plt.xticks(range(1,len(effect)+1), details.index.tolist())
plt.plot( [0, ]*(len(effect)+2) )
plt.show()

effect


# In[10]:


um_bogo =  why.uplift_model(train_data.copy(),treat='Buy One Get One',control='No Offer')
um_discount = why.uplift_model(train_data.copy(),treat='Discount',control='No Offer', random='RANDOM' ) 

plot_gain(um_bogo,um_discount,normalize=True)


# In[11]:


pd.concat([um_bogo.auuc_score(),um_discount.auuc_score(),],axis=0)


# ## Comparison of SLearner and TLearner and XLearner
# 
# 

# In[12]:


why_s = Why(estimator='slearner',estimator_options=dict(model='lightgbm'))
why_s.fit(train_data,outcome,treatment='offer')

um_s = why_s.uplift_model(train_data.copy(),treat='Buy One Get One',control='No Offer', name='SLearner')


# In[13]:


why_t = Why(estimator='tlearner',estimator_options=dict(model='lightgbm'))
why_t.fit(train_data,outcome,treatment='offer')

um_t = why_t.uplift_model(train_data.copy(),treat='Buy One Get One',control='No Offer', name='TLearner')


# In[14]:


why_x = Why(estimator='xlearner',estimator_options=dict(model='lightgbm'))
why_x.fit(train_data,outcome,treatment='offer')

um_x = why_x.uplift_model(train_data.copy(),treat='Buy One Get One',control='No Offer', name='XLearner', random='RANDOM')


# In[15]:


plot_gain(um_s,um_t,um_x,normalize=True)


# In[16]:


pd.concat([um_s.auuc_score(),um_t.auuc_score(),um_x.auuc_score()],axis=0)


# In[ ]:




