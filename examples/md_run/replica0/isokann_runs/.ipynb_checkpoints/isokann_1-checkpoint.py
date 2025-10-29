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
read_dirs_paths('../../dir_paths_.txt', globals())

device = pt.device("cuda" if pt.cuda.is_available() else "cpu")
print("")
print(device)




# Load initial and final states and convert to torch
# D0 = pt.load(out_trajectories1 + 'PWDistances_0_half.pt', map_location=device)
# DT = pt.load(out_trajectories1 + 'PWDistances_t_half.pt', map_location=device)

D0 = pt.load(out_trajectories1 + 'PWDistances_0_full_1.pt', map_location=device)
Dt = pt.load(out_trajectories1 + 'PWDistances_t_full_frame_9.pt', map_location=device)


x0_mean = D0.mean(dim=-1, keepdim=True) 
x0_std = D0.std(dim=-1, keepdim=True)
D0_n = abs(D0 - x0_mean) / (x0_std + 1e-8)

# frame = 9

# Dt = DT[frame,:,:,:]

xt_mean = Dt.mean(dim=-1, keepdim=True) 
xt_std = Dt.std(dim=-1, keepdim=True)
Dt_n = abs(Dt - xt_mean) / (xt_std + 1e-8)



Npoints = D0.shape[0]
Ndims   = D0.shape[1]

print(Dt_n.shape)
print(D0_n.shape)



#import pickle 

    
# Load from the file
#with open(out_isokann + 'hyperparameters_final_40f.pkl', 'rb') as file:
#    best_hyperparams = pickle.load(file)



#print("The best hyperparameters are:", best_hyperparams)



# In[5]:
# best_hyperparams = { 
#                     'Nepochs': 20, 
#                     'nodes': np.array([46056, 23028, 11514, 5757, 2878, 1439, 1]), 
#                     'learning_rate':0.0001, 
#                     'weight_decay':3.727593720314938e-06, 
#                     'batch_size':200, 
#                     'momentum':0.6, 
#                     'patience':2, 
#                     'act_fun': 'leakyrelu' 
# }

# best_hyperparams = {
#                     'Nepochs': 20,
#                     'nodes': np.array([46056, 16384, 8192, 4096, 1]),
#                     'learning_rate':9e-4,
#                     'weight_decay':5e-6,
#                     'batch_size':200,
#                     'momentum':0.8,
#                     'patience':8,
#                     'act_fun': 'sigmoid'
#     }


# =============================================================================
# best_hyperparams = {
#                     'Nepochs': 80,
#                     'nodes': np.array([46056, 4096, 1024, 256, 1]),
#                     'learning_rate':0.001,
#                     'weight_decay':1e-5,
#                     'batch_size':256,
#                     'momentum':0.9,
#                     'patience':10,
#                     'act_fun': 'leakyrelu'
#     }
# 
# =============================================================================

best_hyperparams = {
                    'Nepochs': 130,
                    'nodes': np.array([46056,  4096,  1024,   256,   1]),
                    'learning_rate':0.004,
                    'weight_decay':3.5000000000000004e-05,
                    'batch_size':448,
                    'momentum':0.9199999999999999,
                    'patience':8,
                    'act_fun': 'leakyrelu'
    }

# Power method iterations
Niters    = 400

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
train_LOSS, val_LOSS, best_loss, convergence, xtrema = power_method(D0_n,
                                                            Dt_n,
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
    chi  = f_NN(D0_n).cpu().detach().numpy()


# In[5]:


#print('Correlation with end-to-end distance:', np.corrcoef(R0, chi)[0,1])



# In[6]:

out_folder = get_latest(out_isokann, iso_outs=True, create_new=True)
np.savetxt(os.path.join(out_folder,'val_LOSS_lr.txt'), val_LOSS)
np.savetxt(os.path.join(out_folder,'train_LOSS_lr.txt'), train_LOSS)
np.savetxt(os.path.join(out_folder,'chi0_lr.txt'), chi)
np.savetxt(os.path.join(out_folder,'convergence_lr.txt'), convergence)
np.savetxt(os.path.join(out_folder,'chi_extrema_lr.txt'), xtrema)


# Calculate propagated chi
#chit = f_NN(Xtau).cpu().detach().numpy()
#np.save(out_isokann + 'chit.npy', chit)


# In[ ]:




