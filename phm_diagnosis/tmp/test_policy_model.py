#!/usr/bin/env python
# coding: utf-8

# In[37]:


import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath('./introduction.ipynb'))))


# In[38]:


import numpy as np

from ylearn.policy.policy_model import PolicyTree
from ylearn.utils._common import to_df
import matplotlib.pyplot as plt


# In[39]:


v = np.random.normal(size=(1000, 10))
y = np.hstack([v[:, [0]] < 0, v[:, [0]] > 0])

data = to_df(v=v)
covariate = data.columns


# In[40]:


est = PolicyTree(criterion='policy_reg', max_depth=2, min_impurity_decrease=0.01)
est.fit(data=data, covariate=covariate, effect_array=y.astype(float))


# In[41]:


est.plot()
plt.show()


# In[42]:


est.decision_path(v=v[1, :].reshape(1, -1))


# In[43]:


est.decision_path(data=data[:1])


# In[44]:


est1 = PolicyTree(criterion='policy_test')
est1.fit(data, covariate, effect_array=y)


# In[45]:


est2 = PolicyTree(criterion='policy_test1')
est2.fit(data=data, covariate=covariate, effect_array=y)

