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
sys.path.append(os.path.abspath('../../../'))

from src.useful_functions import read_dirs_paths
from src.isokann.modules4 import *

# For matplotlib
font = {'size'   : 10}
plt.rc('font', **font)
in2cm = 1/2.54  # centimeters in inches

# For reproducibility
np.random.seed(0)
pt.manual_seed(0)

# Read directory paths
read_dirs_paths('dir_paths.txt', globals())

device = pt.device("cuda" if pt.cuda.is_available() else "cpu")
print("")
print(device)


# In[2]:


# Load initial and final states and convert to torch
D0 = pt.load(out_trajectories1 + 'PWDistances_0.pt', map_location=device)
DT = pt.load(out_trajectories1 + 'PWDistances_t.pt', map_location=device)

Npoints = D0.shape[0]
Ndims   = D0.shape[1]

Nfinpoints  = DT.shape[1]
Nframes     = DT.shape[3]


frame = 0
# Dt = pt.clone(DT[frame,:,:,:])
Dt = DT[frame,:,:,:]


print(Dt.shape)


# In[3]:


NN_epochs = [5, 10 ,15]


NN_nodes =          [(Ndims, int(2*Ndims/3), 1), 
                      (Ndims, int(2*Ndims/3), int(Ndims/3), 1),
                      (Ndims, int(Ndims/2), 1), 
                      (Ndims, int(Ndims/2),int(Ndims/4), 1)]

NN_lr = [ 0.001,
          0.0005,
          0.0001]

NN_wd  = [ 0.001,
           0.0007,
           0.0003]

NN_bs  = [100, 200, 500]

NN_patience = [2, 3, 5]

NN_act_fun = ['sigmoid']

best_hyperparams, best_val_loss  = random_search(D0,
                                                Dt,
                                                NN_epochs,
                                                NN_nodes,
                                                NN_lr,
                                                NN_wd,
                                                NN_bs,
                                                NN_patience,
                                                NN_act_fun,
                                                search_iterations=40,
                                                test_size = 0.2,
                                                out_dir = out_isokann)

print("The best hyperparameters are:", best_hyperparams)
print("The best validation loss is:",  best_val_loss)


# In[4]:


import pickle
# Save to a file
with open(out_isokann + 'hyperparameters_final.pkl', 'wb') as file:
    pickle.dump(best_hyperparams, file)

