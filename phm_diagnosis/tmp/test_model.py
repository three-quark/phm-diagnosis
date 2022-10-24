#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath('./introduction.ipynb'))))


# In[2]:


import networkx as nx
from ylearn.causal_model import model, graph


# ## The first test example of the id method. 
# 
# ### Initialization.

# In[3]:


# Construct the causal graph for the causal model
causation = {
    'X': ['W'],
    'Z': ['X', 'W'],
    'W': [],
    'Y': ['Z'],
}
arc = [('X', 'Y')]
cg = graph.CausalGraph(causation, latent_confounding_arcs=arc)


# In[4]:


cm = model.CausalModel(causal_graph=cg)


# ### id method.

# In[5]:


# Test the identification method
p = cm.id({'Y'}, {'X'})


# In[6]:


causation = {
    'X': ['W'],
    'Z': ['X', 'W'],
    'W': [],
    'Y': ['Z'],
}
arc = [('X', 'Y')]
cg = graph.CausalGraph(causation, latent_confounding_arcs=arc)
p = cm.id({'Y'}, {'X'})
p.show_latex_expression()


# In[7]:


print(p.parse())


# ### Final result.
# 
# The above expressions are combined to give the final result:
# 
# \begin{equation}
# P(y|do(x)) = \sum_{w, z} P(w)P(z|x, w)\sum_{x} P(x|w)P(y|x,w,z)
# \end{equation}
# 
# Exactly the desired one.

# ## The second test example of the id method.

# In[8]:


causation1 = {
    'X': ['Z1'],
    'Z1': [],
    'Z2': ['X'],
    'Y': ['Z2'],
}
arcs1 = [('X', 'Z1'), ('Z1', 'Z2'), ('Z1', 'Y'), ('X', 'Y')]
cg1 = graph.CausalGraph(causation1, latent_confounding_arcs=arcs1)
cm1 = model.CausalModel(cg1)


# In[9]:


cm1.id({'Y'}, {'X'})


# #### The causal effect in the above example can not be identified. The id method gives us exactly what we want.
# 
# ### Test another one.

# In[3]:


from ylearn.causal_model.model import CausalModel
causation5 = {
    'X': ['Z2'],
    'Y': ['Z1', 'Z3'],
    'Z1': ['X', 'Z2'],
    'Z2': [],
    'Z3': ['Z2']
}
arcs5 = [('X', 'Z2'), ('X', 'Y'), ('X', 'Z3'), ('Z2', 'Y')]
cg5 = graph.CausalGraph(causation5, latent_confounding_arcs=arcs5)
cm5 = model.CausalModel(cg5)
p5 = cm5.id({'Y'}, {'X'})


# In[4]:


cm = CausalModel(cg5)


# In[7]:


stat_estimand = cm.id(y={'Y'}, x={'X'})
stat_estimand.show_latex_expression()


# In[11]:


p5.show_latex_expression()


# #### The next example can be done with the frontdoor adjustment, we check wether the result returned by the id method will conincide with that returned by the frontdoor adjustment.

# In[12]:


causation6 = {
    'X': [],
    'Y': ['Z'],
    'Z': ['X'],
}
arcs6 = [('X', 'Y')]
cg6 = graph.CausalGraph(causation6, latent_confounding_arcs=arcs6)
cm6 = model.CausalModel(cg6)
p6 = cm6.id('Y', 'X')
p6


# In[13]:


p6.show_latex_expression()


# In[14]:


p.marginal


# In[15]:


print(p.parse())


# $\sum_{Z}\left[\left[P(Z|X)\right]\right]\left[\sum_{X}\left[P(Y|X, Z)\right]\left[P(X)\right]\right]$

# The above result will be converted to the following probability expression
# 
# \begin{align*}
#     P(y|do(x)) & = \sum_{z} \sum_x P(x)P(y|x, z) P(z|x)\\
#                 & = \sum_{z} P(y|z) P(z|x)
# \end{align*}
# This is correct.

# ## Now we run one more example.

# In[17]:


causation2 = {
    'W1': [],
    'W2': [],
    'X': ['W1'],
    'Y1': ['X'],
    'Y2': ['W2']
}
arcs2 = [('W1', 'Y1'), ('W1', 'W2'), ('W1', 'Y2'), ('W1', 'Y1')]
cg2 = graph.CausalGraph(causation2, latent_confounding_arcs=arcs2)
cm2 = model.CausalModel(cg2)


# In[18]:


p2 = cm2.id({'Y1', 'Y2'}, {'X'})


# In[19]:


p2.show_latex_expression()


# In[20]:


print(p2.parse())


# $\sum_{W2}\left[P(W2)\right]\left[\sum_{W1}\left[P(Y1|X, W1)\right]\left[P(W1)\right]\right]\left[\left[P(Y2|W2)\right]\right]$

# In[22]:


cm2.causal_graph.remove_incoming_edges({'X'}, new=True).ancestors({'Y1', 'Y2'})


# In[23]:


list(cm2.causal_graph.remove_nodes({'X', 'W1'}, new=True).c_components)


# The above result gives us the following expression
# 
# \begin{align*}
#     P(y_1, y_2 |do(x)) &= \sum_{w_2} P(y_2|w_2)\sum_{w_1}P(w_1)P(y_1|w_1, x)P(w_2) \\
#                         & = \sum_{w_2}P(y_2, w_2)\sum_{w_1}P(w_1)P(y_1|w_1, x)
# \end{align*}
# 
# Also correct.

# # We now test other methods.
# 
# ## Start from the backdoor.

# In[8]:


import networkx as nx
from ylearn.causal_model import model, graph
causation3 = {
    'X1': [],
    'X2': [],
    'X3': ['X1'],
    'X4': ['X1', 'X2'],
    'X5': ['X2'],
    'X6': ['X'],
    'X': ['X3', 'X4'],
    'Y': ['X6', 'X4', 'X5', 'X']
}
cg3 = graph.CausalGraph(causation3)
cm3 = model.CausalModel(cg3)


# In[19]:


backdoor_set, prob = cm3.identify(treatment={'X'}, outcome={'Y'}, identify_method=('backdoor', 'simple'))['backdoor']


# In[20]:


print(backdoor_set)


# In[9]:


ad, p3 = list(cm3.identify({'X'}, {'Y'}, identify_method=('backdoor', 'simple')).values())[0]


# In[10]:


p3.show_latex_expression()


# In[28]:


cm3.identify({'X'}, {'Y'}, identify_method=('backdoor', 'all'))


# In[29]:


cm3.identify({'X'}, {'Y'}, identify_method=('backdoor', 'minimal'))


# ### Verified. Now test other methods related to backdoor adjustment.

# In[30]:


cm3.is_valid_backdoor_set({'X1', 'X4'}, {'X'}, {'Y'})


# In[31]:


cm3.is_valid_backdoor_set({'X4'}, 'X', 'Y')


# In[32]:


cm3.get_backdoor_path('X', 'Y')


# In[33]:


cm3.has_collider(['X', 'X3', 'X1', 'X4', 'X2', 'X5', 'Y'])


# In[34]:


cm3.is_connected_backdoor_path(['X', 'X4', 'X2', 'X5', 'Y'])


# ## Now test methods related to frontdoor adjustment.

# In[35]:


causation4 = {
    'X': [],
    'Z': ['X'],
    'Y': ['Z']
}
arcs4 = [('X', 'Y')]
cg4 = graph.CausalGraph(causation4, latent_confounding_arcs=arcs4)
cm4 = model.CausalModel(cg4)


# In[36]:


cm4.get_backdoor_path('X', 'Z')


# In[38]:


cm4.has_collider(['Z', 'X', 'U0', 'Y'])


# In[40]:


cm4.is_connected_backdoor_path(['Z', 'X', 'U0', 'Y'])


# In[41]:


cm4.is_valid_backdoor_set({'X'}, {'Z'}, {'Y'})


# In[42]:


cm4.is_frontdoor_set({'Z'}, 'X', 'Y')


# In[43]:


z, p = cm4.get_frontdoor_set('X', 'Y')


# In[44]:


p.show_latex_expression()


# In[45]:


print(p.parse())


# $\sum_{Z}\left[P(Z|X)\right]\left[\sum_{X}\left[P(Y|Z, X)\right]\left[P(X)\right]\right]$

# In[46]:


p.show_latex_expression()


# ## Let's see if our tool can be used in a previous paper (https://arxiv.org/pdf/2009.13000.pdf). 

# In[47]:


cau = {
    'D': [],
    'C': ['D', 'X'],
    'X': ['D'],
    'Y': ['C', 'X']
}
cgg = graph.CausalGraph(causation=cau)
cmm = model.CausalModel(cgg)


# In[48]:


ppp = cmm.id({'Y'}, {'X'})


# In[49]:


ppp.show_latex_expression()


# ## Exactly what we want.

# ### Now we test methods related to iv

# In[22]:


causation7 = {
    'p':[],
    't': ['p'],
    'l': ['p'],
    'g': ['t', 'l']
}
arc7 = [('t', 'g')]
cg7 = graph.CausalGraph(causation=causation7, latent_confounding_arcs=arc7)
cm7 = model.CausalModel(causal_graph=cg7)


# In[23]:


cm7.get_iv('t', 'g')


# In[24]:


causation8 = {
    'p':[],
    't': ['p', 'l'],
    'l': [],
    'g': ['t', 'l']
}
cg8 = graph.CausalGraph(causation=causation8, latent_confounding_arcs=arc7)
cm8 = model.CausalModel(causal_graph=cg8)
cm8.get_iv('t', 'g')

