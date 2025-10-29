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

from src.useful_functions import *
from src.isokann.modules3_3 import *

# For matplotlib
font = {'size'   : 10}
plt.rc('font', **font)
in2cm = 1/2.54  # centimeters in inches

# For reproducibility
np.random.seed(0)
pt.manual_seed(0)

# Read directory paths
read_dirs_paths('../dir_paths_.txt', globals())

device = pt.device("cuda" if pt.cuda.is_available() else "cpu")
print("")
print(device)


# In[2]:


# Load initial and final states and convert to torch
D0 = pt.load(out_trajectories1 + 'PWDistances_0_full.pt', map_location=device)
Dt = pt.load(out_trajectories1 + 'PWDistances_t_1_full_frame_9.pt', map_location=device)

Ndims   = D0.shape[1]


print(Dt.shape)
print(D0.shape)


x0_mean = D0.mean(dim=-1, keepdim=True) 
x0_std = D0.std(dim=-1, keepdim=True)
D0_n = abs(D0 - x0_mean) / (x0_std + 1e-8)


xt_mean = Dt.mean(dim=-1, keepdim=True) 
xt_std = Dt.std(dim=-1, keepdim=True)
Dt_n = abs(Dt - xt_mean) / (xt_std + 1e-8)




# In[3]:




NN_nodes =           [(46056, 4096, 1024, 256, 1),
                      (46056, 2048, 512, 1),
                      (Ndims, int(Ndims/6), int(Ndims/12), 1), 
                      (Ndims, int(Ndims/4), int(Ndims/8), 1),
                      (46056, 4096, 1), 
                      (46056, 512, 1)]

NN_lr = np.linspace(5e-4, 5e-3, 10)        
NN_wd = np.linspace(9e-6, 1e-4, 8)         
NN_bs = np.arange(128, 512, 64)           
NN_patience = np.arange(5, 10,1)              
NN_epochs = np.arange(60, 150, 10)            
NN_mu = np.linspace(0.85, 0.99, 5)
NN_act_fun = ['leakyrelu', 'gelu', 'sigmoid', 'relu', 'tanh']
Niters = 200
# NN_lr = np.linspace(5e-4, 1e-3, 10)        
# NN_wd = np.linspace(1e-6, 1e-3, 8)         
# NN_bs = np.arange(100, 600, 100)           
# NN_patience = np.arange(2, 8)              
# NN_epochs = np.arange(100, 150, 10)            
# NN_mu = np.linspace(0.6, 0.99, 5)
# NN_act_fun = ['sigmoid']
# Niters = 100
best_hyperparams, best_val_loss  = random_search(D0_n,
                                                Dt_n,
                                                NN_epochs,
                                                NN_nodes,
                                                NN_lr,
                                                NN_wd,
                                                NN_bs,
                                                NN_mu,
                                                NN_patience,
                                                NN_act_fun,
                                                Niters = Niters,
                                                search_iterations=50,
                                                test_size = 0.2,
                                                out_dir = out_isokann)

print("The best hyperparameters are:", best_hyperparams)
print("The best validation loss is:",  best_val_loss)


# In[4]:


import pickle
# Save to a file
out_folder = get_latest(os.path.join(out_isokann, 'rs_outputs/'), iso_outs=True, create_new=True)
with open(out_folder + 'hyperparameters_final_40f_2.pkl', 'wb') as file:
    pickle.dump(best_hyperparams, file)


