#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath('./introduction.ipynb'))))


# In[4]:


get_ipython().system('pip install igraph -i https://pypi.douban.com/simple')


# In[70]:


import pandas as pd

from ylearn.exp_dataset.gen import gen
from ylearn.causal_discovery import CausalDiscovery


# In[71]:


gen().shape


# In[72]:


X = gen()
X = pd.DataFrame(X, columns=[f'x{i}' for i in range(X.shape[1])])


# In[73]:


X


# In[148]:


import numpy as np
df = pd.read_csv('data/BankChurners.csv.zip')
cols = ['Customer_Age', 'Gender', 'Dependent_count', 'Education_Level', 'Marital_Status', 'Income_Category', 
        'Months_on_book', 'Card_Category', 'Credit_Limit',  
        'Total_Trans_Amt' 
     ]
cols = ['Customer_Age', 'Total_Trans_Amt', 'Credit_Limit','Dependent_count','Months_on_book']

data = df[cols]
data.head().T

data = data.astype(np.float16)
data.shape


# In[150]:


data


# In[151]:


data.dropna()


# In[152]:


data.iloc[:3,:]


# In[153]:


def mean_norm(df_input):
    return df_input.apply(lambda x: ((x.max()-x) / (x.max()-x.min())), axis=0)
X = mean_norm(data.iloc[:,:])
print(X)
import pdb
pdb.set_trace()

# In[154]:


cd = CausalDiscovery(hidden_layer_dim=[2])
est = cd(X, threshold=0.01, return_dict=True)


# In[155]:


print(est)


# In[156]:


import ylearn
ylearn.causal_model.graph.CausalGraph(ylearn.causal_model.model.CausalModel(est).causal_graph).plot()


# In[ ]:


why.causal_effect(test_data, control=['M','College'])


# In[ ]:




