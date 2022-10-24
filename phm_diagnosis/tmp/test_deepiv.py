#!/usr/bin/env python
# coding: utf-8

# In[4]:


import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath('./introduction.ipynb'))))


# In[5]:


import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader


from ylearn.estimator_model.deepiv import Net, NetWrapper, DeepIV, MixtureDensityNetwork, MDNWrapper
from ylearn.estimator_model.utils import BatchData, DiscreteOBatchData


# ## We divide this notebook into 3 sections, where we
# 1. ### test Net and NetWrapper in the first section
# 2. ### test MDN and MDNWrapper in the second section
# 3. ### test deepiv in the final section.

# ### Section 1
# 
# - continuous input, continuous output

# In[6]:


net = Net(x_d=1, w_d=1, out_d=1)
net_wrapped = NetWrapper(net, is_y_net=False)
x = torch.normal(0, 1, size=(1000, 1))
w = torch.ones(1000, 1)

def f(x, w):
    return x * x + torch.exp(x) + 3 * w

target = f(x, w)

net_wrapped.fit(
    x=x,
    w=w,
    target=target,
    device='cpu',
    epoch=10,
)
((net_wrapped.predict(x, w) - f(x, w)) / f(x, w)).mean(dim=0)


# - contiuous input, discrete output

# In[7]:


out_d = 5
net = Net(x_d=1, w_d=1, out_d=out_d, is_discrete_output=True)
sm = nn.Softmax(dim=1)
loss = nn.CrossEntropyLoss()
result = sm(net(torch.ones(5, 1), torch.randn(5, 1)))
l = loss(result, torch.eye(5, 5))
l.backward()


# In[8]:


x = torch.normal(0, 1, size=(1000, 1))
w = torch.normal(1, 2, size=(1000, 1))

def f(x, w):
    xw = torch.cat((x, w), dim=1)
    weight = torch.normal(0, 1, size=(xw.shape[1], 1))
    label_sign = torch.einsum('nd,dc->nc', [xw, weight])
    label = (label_sign > 0).to(int).squeeze()
    return F.one_hot(label)

target = f(x, w)

net = Net(1, 1, 2, is_discrete_output=True)
net_wrapped = NetWrapper(net=net, is_y_net=False)
net_wrapped.fit(x, w, target=target, device='cpu', epoch=20)
# training loss
nn.NLLLoss()(torch.log(net_wrapped.predict_proba(x, w)), torch.argmax(f(x, w), dim=1))


# - discrete input, discret output

# In[9]:


net = Net(x_d=3, w_d=1, out_d=5, is_discrete_input=True, is_discrete_output=True)
sm = nn.Softmax(dim=1)
loss = nn.CrossEntropyLoss()
x_input = torch.randint(0, 3, size=(5,)).squeeze()
w_input = torch.ones(5, 1)
result = sm(net(x_input, w_input))
l = loss(result, torch.eye(5, 5))
l.backward()


# In[10]:


x = torch.eye(1000, 5).index_select(dim=0, index=torch.randint(0, 5, size=(1000,)))
w = torch.normal(0, 1, size=(1000, 1))

def f(x, w):
    xw = torch.cat((x, w), dim=1)
    weight = torch.normal(0, 1, size=(xw.shape[1], 1))
    label_sign = torch.einsum('nd,dc->nc', [xw, weight])
    label = (label_sign > 0).to(int).squeeze()
    return F.one_hot(label)

target = f(x, w)

net = Net(x_d=5, w_d=1, out_d=2, is_discrete_output=True, is_discrete_input=True)
net_wrapped = NetWrapper(net)
net_wrapped.fit(x, w, target=target, device='cpu', epoch=10)
nn.NLLLoss()(torch.log(net_wrapped.predict_proba(x, w)), torch.argmax(f(x, w), dim=1))


# - discrete input, continuous output

# In[11]:


net = Net(x_d=3, w_d=1, out_d=1, is_discrete_input=True, is_discrete_output=False)
sm = nn.Softmax(dim=1)
loss = nn.MSELoss()
x_input = torch.randint(0, 3, size=(5,)).squeeze()
w_input = torch.ones(5, 1)
result = sm(net(x_input, w_input))
l = loss(result, torch.ones(5, 1))
l.backward()


# In[12]:


x = torch.eye(1000, 5).index_select(dim=0, index=torch.randint(0, 5, size=(1000,)))
w = torch.normal(0, 2, size=(1000, 1))

def f(x, w):
    weight = torch.normal(0, 1, size=(6, 1))
    xw = torch.cat((x, w), dim=1)
    target = torch.einsum('nd,dc->nc', [xw, weight])
    return target

target = f(x, w)

net = Net(x_d=5, w_d=1, out_d=1, is_discrete_input=True)
net_wrapped = NetWrapper(net)
net_wrapped.fit(x=x, w=w, target=target, device='cpu', epoch=10)
nn.MSELoss()(net_wrapped.predict(x=x, w=w), target)


# ### Section 2
# 
# Test the MDN and MDNWrapper

# In[13]:


z_d = 3
w_d = 2
out_d = 1
num_gaussian = 10
mdn = MixtureDensityNetwork(
    z_d=z_d,
    w_d=w_d,
    out_d=out_d,
    num_gaussian=num_gaussian,
    is_discrete_input=False,
)
mdn_wrapped = MDNWrapper(mdn=mdn)


# In[14]:


n = 200
z = torch.normal(1, 2, size=(n, z_d))
w = torch.normal(0, 1, size=(n, w_d))
zw = torch.cat((z, w), dim=1)
weight = torch.normal(1, 2, size=(zw.shape[1], 1))
target = torch.distributions.Poisson(torch.exp(zw.matmul(weight))).sample()


# In[15]:


# mdn_wrapped.fit(
#     z=z,
#     w=w,
#     target=target,
#     device='cpu',
#     lr=0.2,
#     batch_size=20,
#     epoch=2,
# )


# In[16]:


import numpy as np
import matplotlib.pyplot as plt

n = 5000

# Initialize exogenous variables; normal errors, uniformly distributed covariates and instruments
e = np.random.normal(size=(n, 1))
w = np.random.uniform(low=0.0, high=10.0, size=(n, 1))
z = np.random.uniform(low=0.0, high=10.0, size=(n, 1))

e, w, z = torch.tensor(e), torch.tensor(w), torch.tensor(z)
weight_w = torch.randn(1)
weight_z = torch.randn(1)

def treatment(w, z, e):
    x = torch.sqrt(w) * weight_w + torch.sqrt(z) * weight_z + e
    x = (torch.sign(x) + 1) /  2
    return F.one_hot(x.reshape(-1).to(int))


# In[17]:


# Outcome equation 
weight_x = torch.randn(2, 1)
weight_wx = torch.randn(2, 1)
def outcome(w, e, treatment):
    wx = torch.mm(treatment.to(torch.float32), weight_x)
    wx1 = (w * treatment.to(torch.float32)).to(torch.float32).matmul(weight_wx.to(torch.float32))
    # wx1 = w
    return (wx**2) * 10 - wx1 + e / 2
treatment = treatment(w, z, e)
y = outcome(w, e, treatment)


# In[18]:


data_dict = {
    'z': z.squeeze().to(torch.float32),
    'w': w.squeeze().to(torch.float32),
    'x': torch.argmax(treatment, dim=1),
    'y': y.squeeze().to(torch.float32)
}
data = pd.DataFrame(data_dict)
data.head()


# In[19]:


iv = DeepIV(is_discrete_treatment=True)
iv.fit(
    data=data,
    outcome='y',
    treatment='x',
    instrument='z',
    adjustment='w',
    device='cpu',
    batch_size=2500,
    lr=0.5,
    epoch=1,
)


# In[20]:


iv.estimate()


# In[21]:


n = 5000

# Initialize exogenous variables; normal errors, uniformly distributed covariates and instruments
e = np.random.normal(size=(n,))
x = np.random.uniform(low=0.0, high=10.0, size=(n,))
z = np.random.uniform(low=0.0, high=10.0, size=(n,))

# Initialize treatment variable
t = np.sqrt((x+2) * z) + e

# Show the marginal distribution of t
plt.hist(t)
plt.xlabel("t")
plt.show()

plt.scatter(z[x < 1], t[x < 1], label='low X')
plt.scatter(z[(x > 4.5) * (x < 5.5)], t[(x > 4.5) * (x < 5.5)], label='moderate X')
plt.scatter(z[x > 9], t[x > 9], label='high X')
plt.legend()
plt.xlabel("z")
plt.ylabel("t")
plt.show()

# Outcome equation 
y = t*t / 10 - x*t / 10 + e

# The endogeneity problem is clear, the latent error enters both treatment and outcome equally
plt.scatter(t,z, label ='raw data')
tticks = np.arange(-2,12)
yticks2 = tticks*tticks/10 - 0.2 * tticks
yticks5 = tticks*tticks/10 - 0.5 * tticks
yticks8 = tticks*tticks/10 - 0.8 * tticks
plt.plot(tticks,yticks2, 'r--', label = 'truth, x=2')
plt.plot(tticks,yticks5, 'g--', label = 'truth, x=5')
plt.plot(tticks,yticks8, 'y--', label = 'truth, x=8')
plt.xlabel("t")
plt.ylabel("y")
plt.legend()
plt.show()


# In[23]:


import pandas as pd
from ylearn.estimator_model.utils import convert2tensor
z, w, x, y = convert2tensor(z, x, t, y)
z = z.squeeze().to(torch.float32)
w = w.squeeze().to(torch.float32)
x = x.squeeze().to(torch.float32)
y = y.squeeze().to(torch.float32)

data_dict = {
    'z': z,
    'w': x,
    'x': t,
    'y': y
}
data = pd.DataFrame(data_dict)


# In[24]:


iv = DeepIV(num_gaussian=10)
iv.fit(
    data=data,
    outcome='y',
    treatment='x',
    instrument='z',
    adjustment='w',
    sample_n=2,
    lr=0.5,
    epoch=1,
    device='cpu',
    batch_size=5000
)


# In[25]:


iv.estimate()

