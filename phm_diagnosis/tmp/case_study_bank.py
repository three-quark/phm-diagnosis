#!/usr/bin/env python
# coding: utf-8

# # YLearn Case Study - Bank Customer Transaction Amount

# ## Content
# 
# 1. Summary
# 1. Dataset
# 1. Causal Effect with Defined Treatment
#     1. Statistics of Card_Category
#     1. Learn Why with `treatment='Card_Category'`
#     1. Estimate causal effect
#     1. Counterfactual inference
#     1. Policy interpreter
#     1. Effect comparison
#     1. Conclusion
# 1. Causal Effect with Discovery
#     1. Learn Why with `treatment=None`
#     1. Estimate causal effect on train_data
#     1. Estimate causal effect on test_data
#     1. Conclusion

# ## 1. Summary
# 
# This notebook utilizes a typical bank customer dataset to demonstrate the usage of all-in-one API `Why` of YLearn. `Why` covers the full processing pipeline of causal learning, including causal discovery, causal effect identification, causal effect estimation, counterfactual inference, and policy learning. 
# 
# One should firstly create an instance of `Why` which needs to be trained by calling its method `fit()`. Both customized setting (with a defined treatment) and default setting (to discover treatments by the algorithm itself) are introduced. The following utilities, such as `plot_causal_graph()`, `causal_effect()` and `whatif()` are performed to analyze various counterfactual scenarios. Another key method `policy_interpreter()` provides a customized solution to optimize the output. 
# 

# ## 2. Dataset
# 
# The dataset used in this notebook is a subset of Kaggle [BankChurners](https://www.kaggle.com/datasets/syviaw/bankchurners) dataset. It’s intended to predict churning customers. However, we use it to analyze the causes of total transaction amount and provide customized strategies to maximize the transaction amount. The dataset contains the information of around 10,000 customers with nearly 20 continuous and categorical variables that represent user's characteristics. We manually select 10 variables to analyze their causal relations and causal effect, which are shown below:
# 
# * Customer general features: Customer_Age, Gender, Dependent_count, Education_Level, Marital_Status,  Income_Category
# * Customer credit card features: Months_on_book, Card_Category, Credit_Limit
# * Customer transaction amount: Total_Trans_Amt
# 
# The `Total_Trans_Amt` is the outcome in this notebook.
# 

# In[10]:


import numpy as np
import pandas as pd
from matplotlib import pyplot  as plt

from sklearn.model_selection import train_test_split
from ylearn import Why


# In[11]:


import warnings
warnings.filterwarnings('ignore')


# In[12]:


df = pd.read_csv('data/BankChurners.csv.zip')
cols = ['Customer_Age', 'Gender', 'Dependent_count', 'Education_Level', 'Marital_Status', 'Income_Category', 
        'Months_on_book', 'Card_Category', 'Credit_Limit',  
        'Total_Trans_Amt' 
     ]
data = df[cols]
outcome = 'Total_Trans_Amt'

data 


# In[13]:


train_data,test_data=train_test_split(data,test_size=0.3,random_state=123)

print('train_data:',train_data.shape)
print('test_data: ',test_data.shape)


# ## 3. Causal Effect with Defined Treatment
# 

# ### A. Statistics of Card_Category
# 
# Among the three credit card features, `Card_Category` is a categorical and reasonable variable as treatment. An overview of card categories and corresponding average transaction amount is displayed below.

# In[14]:


card_stat=train_data[['Card_Category','Total_Trans_Amt']].groupby('Card_Category').agg(['mean','min','max',]).sort_values(by=('Total_Trans_Amt','mean'))

card_stat[('Total_Trans_Amt','mean')].plot(kind='bar', title='Mean of Total_Trans_Amt')
card_stat


# ### B. Learn Why with `treatment='Card_Category'`
# 
# Firstly, create an instance of `Why`. Then train the model with `fit()` method, which defines `treatment=’Card_Category’.` The printed logs show the information of identified results and used eatimator. To visualize the causal relations, use `plot_causal_graph()` to obtain the causal graph.

# In[16]:


why=Why()
why.fit(train_data,outcome,treatment='Card_Category')


# In[45]:





# In[22]:


test_data.head()


# In[28]:


test_data.iloc[:10,:]


# In[37]:


why.plot_policy_tree(test_data.iloc[:21,:])


# In[53]:


from ylearn.causal_model.model import CausalModel


# In[58]:


why.causal_graph().plot()


# In[72]:


CausalModel(why.causal_graph(), treatment='Card_Category', outcome='Total_Trans_Amt').outcome


# In[38]:


why.plot_causal_graph()


# ### C. Estimate causal effect
# 
# There are four types of card categoties: Blue, Sliver, Gold and Platinum. Taking Blue as control variable and the rest three as treatment variables, method `causal_effect()` outputs three causal effect estimations.  From the results, we find that card upgrade will increase the personal transaction amount. Gold card has the strongest effect.

# In[8]:


effect=why.causal_effect(control='Blue',return_detail=True)
effect=effect.loc['Card_Category'].sort_values(by='mean')
details=effect.pop('detail')

plt.figure(figsize=(10, 5))
plt.violinplot(details.tolist(), showmeans=True)
plt.ylabel('Effect distribution')
plt.xticks(range(1,len(effect)+1), details.index.tolist())
plt.plot( [0, ]*(len(effect)+2) )
plt.show()

effect


# ### D. Counterfactual inference
# 
# As decision makers, they would like to know the estimated increment if upgrading customers' card category. `whatif()` provides the solution to do counterfactual inference.   
# 
# * Upgrade all **Blue** cards to **Sliver**

# In[9]:


whatif_data= train_data[lambda df: df['Card_Category']=='Blue' ]
out_orig=whatif_data[outcome]

value_sliver=whatif_data['Card_Category'].map(lambda _:'Silver')
out_silver=why.whatif(whatif_data,value_sliver,treatment='Card_Category')

print('Selected customers:', len(whatif_data))
print(f'Mean {outcome} with Blue card:\t{out_orig.mean():.3f}' )
print(f'Mean {outcome} if Silver card:\t{out_silver.mean():.3f}' )

plt.figure(figsize=(8, 5), )
out_orig.hist(label='Blue card',bins=30,alpha=0.7)
out_silver.hist(label='Silver card',bins=30,alpha=0.7)
plt.legend()


# * Upgrade all **Blue** cards to **Gold**

# In[10]:


whatif_data= train_data[lambda df: df['Card_Category']=='Blue' ]
out_orig=whatif_data[outcome]

value_gold=whatif_data['Card_Category'].map(lambda _:'Gold')
out_gold=why.whatif(whatif_data,value_gold,treatment='Card_Category')


print('Selected customers:', len(whatif_data))
print(f'Mean {outcome} with Blue card:\t{out_orig.mean():.3f}' )
print(f'Mean {outcome} if Gold card:\t{out_gold.mean():.3f}' )

plt.figure(figsize=(8, 5), )
out_orig.hist(label='Blue card',bins=30,alpha=0.7)
out_gold.hist(label='Gold card',bins=30,alpha=0.7)
plt.legend()


# * Upgrade all **Blue** cards to **Platinum**

# In[75]:


whatif_data= train_data[lambda df: df['Card_Category']=='Blue' ]
out_orig=whatif_data[outcome]

value_platinum=whatif_data['Card_Category'].map(lambda _:'Platinum')
out_platinum=why.whatif(whatif_data,value_platinum,treatment='Card_Category')


print('Selected customers:', len(whatif_data))
print(f'Mean {outcome} with Blue card:\t{out_orig.mean():.3f}' )
print(f'Mean {outcome} if Platinum card:\t{out_platinum.mean():.3f}' )

plt.figure(figsize=(8, 5), )
out_orig.hist(label='Blue card',bins=30,alpha=0.7)
out_platinum.hist(label='Platinum card',bins=30,alpha=0.7)
plt.legend()


# By upgrading all Blue cards to Sliver, Gold and Platinum cards, the estimations of mean Total_Trans_Amt increase from 4231 to 5651, 12477 and 8044 respectively. It's a promising improvement. However, we may wonder if upgrading all Blue to Gold is the optimized solution. 

# ### E. Policy interpreter
# 
# YLearn also provides `policy_interpreter()` method to search the optimized solution.

# In[76]:


whatif_data


# In[80]:


pi=why.policy_interpreter(whatif_data,  max_depth=3)

plt.figure(figsize=(12, 8), )
pi.plot()


# In[81]:


pi.treatment


# The tree intepreter is split into four nodes based on Credit_Limit and Month_on_book. The value array indicates the causal effects of the control variable(Blue vs Blue) and three treatments(Gold vs Blue, Platinum vs Blue, Silver vs Blue). The higher value means the better performance. Looking at the first node, the policy tree recommends to upgrade Blue cards to Gold for customers with 'Credit_limit<=15231.5'. 

# * Upgrade credit card with causal policy

# In[13]:


whatif_data= train_data[lambda df: df['Card_Category']=='Blue' ]
out_orig=whatif_data[outcome]

value_from_policy=pi.decide(whatif_data)
out_from_policy=why.whatif(whatif_data,value_from_policy,treatment='Card_Category')


print('Selected customers:', len(whatif_data))
print(f'Mean of {outcome} with Blue card:\t{out_orig.mean():.3f}' )
print(f'Mean of {outcome} if apply policy:\t{out_from_policy.mean():.3f}' )

plt.figure(figsize=(8, 5), )
out_orig.hist(label='Blue card',bins=30,alpha=0.7)
out_from_policy.hist(label='Card from policy',bins=30,alpha=0.7)
plt.legend()


# ### F. Effect comparison

# In[14]:


whatif_summary=pd.Series({
    'Blue to Silver': out_silver.mean(),
    'Blue to Gold': out_gold.mean(),
    'Blue to Platinum': out_platinum.mean(),
    'Card from Policy': out_from_policy.mean(),
    },name=f'Mean of {outcome}').to_frame()
whatif_summary.plot(kind='bar',legend=False)
whatif_summary


# ### G. Conclusion
# 
# From the comparion results, our policy gets the highest transaction amount! Although the policy doesn't consider costs and risks, it provides the customized strategies to help increase the transaction amount.

# ## 4. Causal effect with Causal Discovery

# ### A. Learn Why with `treatment=None`
# 
# This section uses the default setting of `fit()` method, which means applying causal discovery to find the causal relations. 

# In[15]:


why=Why(random_state=123)
why.fit(train_data,outcome)


# In[16]:


why.plot_causal_graph()


# We observe customer gender and education level are identified as the important factors that impact the total transaction amounts. The two features are quite reasonable. However, it's hard to make them as treaments in practice. 

# ### B. Estimate causal effect on train_data
# 
# Just to show the method usage, we estimate the causal effects by taking 'M'(gender) and 'College'(education level) as treatments. 

# In[17]:


why.causal_effect(control=['M','College'])


# ### C. Estimate causal effect on test_data
# 
# The causal effect estimations of test_data are displayed. 

# In[18]:


why.causal_effect(test_data, control=['M','College'])


# ### D. Conclusion
# 
# Since gender and education level are attribute features, it's not practical to take them as treaments. This section introduces the uasge of causal discovery. Besides, the estimation results from train data and test data are quite similar, which reflects the robustness of the method.  
