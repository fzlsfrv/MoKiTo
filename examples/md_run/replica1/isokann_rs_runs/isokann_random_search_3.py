#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import sys
import os
import torch as pt
from tqdm import tqdm

# Add the project root
sys.path.append(os.path.abspath('../../../../'))

from src.useful_functions import read_dirs_paths
from src.isokann.modules3_2 import *

# For matplotlib
font = {'size'   : 10}
plt.rc('font', **font)
in2cm = 1/2.54  # centimeters in inches

# For reproducibility
#np.random.seed(0)
#pt.manual_seed(0)

# Read directory paths
read_dirs_paths('../dir_paths_.txt', globals())

device = pt.device("cuda" if pt.cuda.is_available() else "cpu")
print("")
print(device)


# In[2]:


# Load initial and final states and convert to torch
D0 = pt.load(out_trajectories1 + 'PWDistances_0_40f-full.pt', map_location=device)
DT = pt.load(out_trajectories1 + 'PWDistances_t_40f-full.pt', map_location=device)


Ndims   = D0.shape[1]



frame = 9

Dt = DT[frame,:,:,:]

print(Dt.shape)
print(D0.shape)


# In[3]:




NN_nodes = [
    (Ndims, int(Ndims//4), int(Ndims//8), int(Ndims//16), int(Ndims//32), 1),
    (Ndims, 4096, 2048, 1024, 256, 1),
    (Ndims, int(Ndims//8), int(Ndims//16), 1),
    (Ndims, int(Ndims//2), int(Ndims//4), int(Ndims//8), 4096, 2048, 256, 1),
    (Ndims, 256, 256, 256, 256, 256, 1),
    (Ndims, 4096, 4096, 4096, 4096, 1)
    
        ]

NN_lr = np.logspace(-3, -2, 10)        
NN_wd = np.logspace(-6, -4, 8)        
NN_bs = np.arange(50, 300, 50)           
NN_patience = np.arange(2, 5)              
NN_epochs = np.arange(50, 100, 10) 
NN_mu = np.linspace(0.6, 0.99, 5)

NN_act_fun = ['leakyrelu']

best_hyperparams, best_val_loss  = random_search(D0,
                                                Dt,
                                                NN_epochs,
                                                NN_nodes,
                                                NN_lr,
                                                NN_wd,
                                                NN_bs,
                                                NN_mu,
                                                NN_patience,
                                                NN_act_fun,
                                                Niters = 100,
                                                search_iterations=50,
                                                test_size = 0.2,
                                                out_dir = out_isokann)

print("The best hyperparameters are:", best_hyperparams)
print("The best validation loss is:",  best_val_loss)


# In[4]:


import pickle
# Save to a file
with open(out_isokann + 'hyperparameters_full.pkl', 'wb') as file:
    pickle.dump(best_hyperparams, file)

