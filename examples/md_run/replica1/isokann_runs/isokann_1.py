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
np.random.seed(0)
pt.manual_seed(0)

# Read directory paths
read_dirs_paths('../dir_paths.txt', globals())

device = pt.device("cuda" if pt.cuda.is_available() else "cpu")
print("")
print(device)


# In[2]:


# Load initial and final states and convert to torch
D0 = pt.load(out_trajectories1 + 'PWDistances_0_aligned.pt', map_location=device)
DT = pt.load(out_trajectories1 + 'PWDistances_t_aligned.pt', map_location=device)

R0 = np.loadtxt(out_trajectories1 + 'R0.txt')


Npoints = D0.shape[0]
Ndims   = D0.shape[1]

print(D0.shape)
print(DT.shape)


frame = 9
Dt = DT[frame,:,:,:]

print(Dt.shape)


import pickle 

    
# Load from the file
with open(out_isokann + 'hyperparameters_final_40f.pkl', 'rb') as file:
    best_hyperparams = pickle.load(file)



print("The best hyperparameters are:", best_hyperparams)

# In[5]:


# Power method iterations
Niters    = 200

# NN hyperparameters
Nepochs   = best_hyperparams['Nepochs']
nodes     = best_hyperparams['nodes']
lr        = best_hyperparams['learning_rate']
wd        = best_hyperparams['weight_decay']
bs        = best_hyperparams['batch_size']
mu        = best_hyperparams['momentum']
patience  = best_hyperparams['patience']
act_fun   = best_hyperparams['act_fun']


tolerance = 0.000001


# Define the interpolating function
f_NN = NeuralNetwork( Nodes = np.asarray(nodes), activation_function = act_fun ).to(device)



# Apply the power method using the best hyperparameters
train_LOSS, val_LOSS, best_loss, convergence = power_method(D0,
                                                            Dt,
                                                            f_NN,
                                                            scale_and_shift,
                                                            Niters = Niters,
                                                            Nepochs = Nepochs,
                                                            tolerance  = tolerance,
                                                            lr = lr,
                                                            wd = wd,
                                                            batch_size = bs,
                                                            momentum = mu,
                                                            patience = patience,
                                                            print_eta  = True,
                                                            test_size = 0.2,
                                                            loss ='full'
                                                                )
with pt.no_grad():    
    chi  = f_NN(D0).cpu().detach().numpy()


# In[5]:


#print('Correlation with end-to-end distance:', np.corrcoef(R0, chi)[0,1])



# In[6]:


np.savetxt(out_isokann + 'outs/val_LOSS_aligned_0_1.txt', val_LOSS)
np.savetxt(out_isokann + 'outs/train_LOSS_aligned_0_1.txt', train_LOSS)
np.savetxt(out_isokann + 'outs/chi0_aligned_0_1.txt', chi)
np.savetxt(out_isokann + 'outs/convergence_aligned_0_1.txt', convergence)


# Calculate propagated chi
#chit = f_NN(Xtau).cpu().detach().numpy()
#np.save(out_isokann + 'chit.npy', chit)


# In[ ]:




