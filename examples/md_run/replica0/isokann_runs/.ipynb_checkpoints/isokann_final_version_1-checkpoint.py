#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import glob
import sys
import os
import torch as pt
from tqdm import tqdm
import gc
# Add the project root
sys.path.append(os.path.abspath('../../../../'))

from src.useful_functions import *
from src.isokann.modules3_2 import *

# For matplotlib
font = {'size'   : 10}
plt.rc('font', **font)
in2cm = 1/2.54  # centimeters in inches

# For reproducibility
np.random.seed(0)
pt.manual_seed(0)

# Read directory paths
read_dirs_paths('../dir_paths_2.txt', globals())

device = pt.device("cuda" if pt.cuda.is_available() else "cpu")
print("")
print(device)


# In[3]:


pw_data_dirs = multiple_dirs(out_trajectories1, fetch_files = False)

# In[4]:


chi = np.empty(6, dtype = object)
train_LOSS = np.empty(6, dtype = object)
val_LOSS = np.empty(6, dtype = object)
best_loss = np.empty(6, dtype = object)
convergence = np.empty(6, dtype = object)
R0_arr = np.empty(6, dtype = object)


# In[45]:


import pickle 

    
# Load from the file
with open(out_isokann + 'hyperparameters_final_40f.pkl', 'rb') as file:
    best_hyperparams = pickle.load(file)



print("The best hyperparameters are:", best_hyperparams)

# In[5]:


# Power method iterations
Niters    = 200

# NN hyperparameters
Nepochs   = 1
nodes     = best_hyperparams['nodes']
lr        = best_hyperparams['learning_rate']
wd        = best_hyperparams['weight_decay']
bs        = best_hyperparams['batch_size']
mu        = best_hyperparams['momentum']
patience  = best_hyperparams['patience']
act_fun   = best_hyperparams['act_fun']


tolerance = 0.000001

niters = 200

# In[ ]:
chk_folder = get_latest(out_isokann, chk_folder=True, create_new=True)
print(chk_folder)

for r in range(niters):

    for i in range(len(pw_data_dirs)):
        
    # Load initial and final states and convert to torch
        D0 = pt.load(pw_data_dirs[i] + 'PWDistances_0_aligned.pt', map_location=device)
        DT = pt.load(pw_data_dirs[i] + 'PWDistances_t_aligned.pt', map_location=device)
        
        print('\n')
        print(f'Getting the {i}th trajectory data from the directory: {pw_data_dirs[i]}...')
    
        R0_arr[i] = np.loadtxt(pw_data_dirs[i] + 'R0.txt')
    
    
        Npoints = D0.shape[0]
        Ndims   = D0.shape[1]
    
    
        frame = 9
        # Dt = pt.clone(DT[frame,:,:,:])
    
        Dt = DT[frame,:,:,:]
    
        print(Dt.shape)
    
    
    
        # Define the interpolating function
        f_NN = NeuralNetwork(Nodes = np.asarray(nodes)).to(device)
    
        chk_folder = get_latest(out_isokann, chk_folder=True)
    
        if i > 0 and os.path.exists(os.path.join(chk_folder,  f"f_NN_chk_{i-1}.pt")):
            print(f'Loading the {i-1}th checkpoint from {chk_folder}...')
            f_NN.load_state_dict(pt.load(os.path.join(chk_folder,  f"f_NN_chk_{i-1}.pt"),  map_location=device))
    
    
    
    
        # Apply the power method using the best hyperparameters
        train_LOSS[i], val_LOSS[i], best_loss[i], convergence[i] = power_method(D0,
                                                                    Dt,
                                                                    f_NN,
                                                                    scale_and_shift,
                                                                    Niters = 1,
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
    
        chi[i]  = f_NN(D0).cpu().detach().numpy()
        
        
        print(f'Saving the {i}th checkpoint into {chk_folder}...')
        
        pt.save(f_NN.state_dict(),os.path.join(chk_folder,  f"f_NN_chk_{i}.pt"))
    
        # 7) FREE big tensors before next replica
        del D0, DT, Dt, f_NN
        gc.collect()
        if device.type == 'cuda':
            pt.cuda.empty_cache()


# In[ ]:

out_folder = get_latest(out_isokann, iso_outs=True, create_new=True)
print(f'Writing the outputs into {out_folder}')
np.save(os.path.join(out_folder, 'val_LOSS.npy'), val_LOSS)
np.save(os.path.join(out_folder, 'train_LOSS.npy'), train_LOSS)
np.save(os.path.join(out_folder,  'chi0.npy'), chi)
np.save(os.path.join(out_folder,  'convergence.npy'), convergence)

# # Calculate propagated chi
# #chit = f_NN(Xtau).cpu().detach().numpy()
# #np.save(out_isokann + 'chit.npy', chit)


# In[ ]:




