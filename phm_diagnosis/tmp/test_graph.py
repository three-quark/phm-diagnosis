#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath('./introduction.ipynb'))))


# In[2]:


from ylearn.causal_model import graph
import networkx as nx
from collections import defaultdict


# In[8]:


from ylearn.causal_model.prob import Prob
var = {'v'}
conditional = {'y'}
marginal = {'w'}
p1 = Prob(variables={'w'}, conditional={'z'})
p2 = Prob(variables={'x'}, conditional={'y'})
p3 = Prob(variables={'u'})
product = {p1, p2, p3}
P = Prob(variables=var, conditional=conditional, marginal=marginal, product=product)
P.show_latex_expression()


# In[9]:


P.parse()


# We first test the initialization. All attributes should be constructed properly.

# In[33]:


causation = defaultdict(list)
causation['X'] = ['Z2']
causation['Z1'] = ['X', 'Z2']
causation['Y'] = ['Z1', 'Z3']
causation['Z3'] = ['Z2']
causation['Z2'] = []
arc = [('X', 'Z2'), ('X', 'Z3'), ('X', 'Y'), ('Z2', 'Y')]
cg = graph.CausalGraph(causation=causation, latent_confounding_arcs=arc)
# cg = graph.CausalGraph(causation=causation)


# In[4]:


cg.causation


# In[34]:


list(cg.dag.predecessors('X'))


# In[6]:


cg.prob


# In[8]:


cg.latent_confounding_arcs


# In[9]:


cg.is_dag


# In[6]:


list(cg.c_components)


# In[11]:


cg.observed_dag.edges


# In[12]:


list(cg.topo_order)


# We now test the methods.

# In[13]:


cg.to_adj_matrix()


# In[14]:


cg.ancestors({'X', 'Z2', 'Z3'})


# In[15]:


# Add an existed node to the CausalGraph.
cg.add_nodes(['X'])
cg.causation


# In[16]:


# Remove an edge
cg.remove_edges_from([('Z2', 'X')])


# In[17]:


# The causation should also be changed accordingly.
cg.causation


# In[18]:


# Add an edge.
cg.add_edges_from([('Z2', 'X')])
cg.causation


# In[19]:


# View all edges, including the observed and unobserved ones.
cg.dag.edges


# In[20]:


# Remove a node. The causation should be changed accordingly.
cg.remove_nodes(['Y'])
cg.causation


# In[21]:


cg.add_edges_from([('Z1', 'Y'), ('Z3', 'Y')])


# In[22]:


cg.causation


# In[23]:


# Create a new CausalGraph and remove the node in it.
cg1 = cg.remove_nodes(['X'], new=True)


# In[24]:


cg1.dag.nodes


# In[25]:


# Build a subgraph with nodes in the subset.
cg2 = cg.build_sub_graph(['X', 'Y', 'Z2'])


# In[26]:


cg2.dag.nodes


# In[27]:


cg2.dag.edges


# In[28]:


# Remove all edges going in to the given set.
# Here all edges with arrow into 'X' should be removed,
# including thoes unobserved confounding arcs.
cg.remove_incoming_edges(['X'])


# In[29]:


cg.dag.edges


# In[30]:


# See if all incoming edges of 'X' have been removed.
cg.dag.in_edges(['X'], keys=True)


# In[31]:


# The causation should be changed accordingly.
cg.causation

