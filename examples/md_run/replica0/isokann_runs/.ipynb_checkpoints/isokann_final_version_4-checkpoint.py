#!/usr/bin/env python
# coding: utf-8

# In[3]:


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
sys.path.append(os.path.abspath('../../../'))

from src.useful_functions import read_dirs_paths
from src.isokann.modules3_1 import *

# For matplotlib
font = {'size'   : 10}
plt.rc('font', **font)
in2cm = 1/2.54  # centimeters in inches

# For reproducibility
np.random.seed(0)
pt.manual_seed(0)

# Read directory paths
read_dirs_paths('dir_paths_2.txt', globals())

device = pt.device("cuda" if pt.cuda.is_available() else "cpu")
print("")
print(device)


# In[3]:


pw_data_dirs = glob.glob(out_trajectories1)
pw_data_dirs = sorted(pw_data_dirs)
print(pw_data_dirs)


# In[4]:


chi = np.empty(6, dtype = object)
train_LOSS = np.empty(6, dtype = object)
val_LOSS = np.empty(6, dtype = object)
best_loss = np.empty(6, dtype = object)
convergence = np.empty(6, dtype = object)
R0_arr = np.empty(6, dtype = object)


# In[38]:


best_hyperparams = { 
                        'Nepochs': np.int64(10), 
                        'nodes': np.array([51040, 34026, 17013,     1]), 
                        'learning_rate': 1e-05, 
                        'weight_decay': 0.0005718571428571429, 
                        'batch_size': np.int64(100), 
                        'patience': np.int64(3), 
                        'act_fun': 'relu'
}

# In[5]:


# Power method iterations
Niters    = 400

# NN hyperparameters
Nepochs   = best_hyperparams['Nepochs']
nodes     = best_hyperparams['nodes']
lr        = best_hyperparams['learning_rate']
wd        = best_hyperparams['weight_decay']
bs        = best_hyperparams['batch_size']
patience  = best_hyperparams['patience']
act_fun   = best_hyperparams['act_fun']


tolerance = 0.000001


# In[ ]:

i = 0
for i in range(len(pw_data_dirs)):
# Load initial and final states and convert to torch
    D0 = pt.load(pw_data_dirs[i] + 'PWDistances_0.pt', map_location=device)
    DT = pt.load(pw_data_dirs[i] + 'PWDistances_t.pt', map_location=device)

    print(f'Getting the {i}th trajectory data from the directory: {pw_data_dirs[i]}')

    R0_arr[i] = np.loadtxt(pw_data_dirs[i] + 'R0.txt')

    
    D0_ = D0[::20,:]
  
    Ndims   = D0_.shape[1]


    frame = 9
    # Dt = pt.clone(DT[frame,:,:,:])

    Dt = DT[frame,::20,:,:]

    print(Dt.shape)



    # Define the interpolating function
    f_NN = NeuralNetwork(Nodes = np.asarray(nodes), activation_function=act_fun).to(device)


    if i > 0 and os.path.exists(out_isokann +  f"nn_chk_4/f_NN_chk_{i-1}.pt"):
        f_NN.load_state_dict(pt.load(out_isokann +  f"nn_chk_4/f_NN_chk_{i-1}.pt", map_location=device))

    # npX0 = D0.cpu().detach().numpy()



    # Apply the power method using the best hyperparameters
    train_LOSS[i], val_LOSS[i], best_loss[i], convergence[i] = power_method(D0_,
                                                                Dt,
                                                                f_NN,
                                                                scale_and_shift,
                                                                Niters = 200,
                                                                Nepochs = Nepochs,
                                                                tolerance  = tolerance,
                                                                lr = lr,
                                                                wd = wd,
                                                                batch_size = bs,
                                                                patience = patience,
                                                                print_eta  = True,
                                                                test_size = 0.2,
                                                                loss ='full'
                                                                )

    chi[i]  = f_NN(D0_).cpu().detach().numpy()

    pt.save(f_NN.state_dict(),out_isokann +  f"nn_chk_4/f_NN_chk_{i}.pt")

    # 7) FREE big tensors before next replica
    del D0_, DT, Dt
    gc.collect()
    if device.type == 'cuda':
        pt.cuda.empty_cache()


# In[ ]:


pt.save(f_NN.state_dict(), out_isokann  + 'nn_chk_4/f_NN.pt')
np.save(out_isokann + 'outs/val_LOSS_4.npy', val_LOSS)
np.save(out_isokann + 'outs/train_LOSS_4.npy', train_LOSS)
np.save(out_isokann + 'outs/chi0_4.npy', chi)
np.save(out_isokann + 'outs/convergence_4.npy', convergence)

# # Calculate propagated chi
# #chit = f_NN(Xtau).cpu().detach().numpy()
# #np.save(out_isokann + 'chit.npy', chit)



