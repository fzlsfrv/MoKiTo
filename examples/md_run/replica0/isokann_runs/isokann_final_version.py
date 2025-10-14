#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from src.isokann.modules3 import *

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


# In[2]:


pw_data_dirs = glob.glob(out_trajectories1)
pw_data_dirs = sorted(pw_data_dirs)
print(pw_data_dirs)


# In[3]:


chi = np.empty(6, dtype = object)
train_LOSS = np.empty(6, dtype = object)
val_LOSS = np.empty(6, dtype = object)
best_loss = np.empty(6, dtype = object)
convergence = np.empty(6, dtype = object)
R0_arr = np.empty(6, dtype = object)


# In[4]:


import pickle 


# Load from the file
with open(out_isokann + 'hyperparameters.pkl', 'rb') as file:
    best_hyperparams = pickle.load(file)


print("The best hyperparameters are:", best_hyperparams)


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


for i in range(len(pw_data_dirs)-1):
# Load initial and final states and convert to torch
    D0 = pt.load(pw_data_dirs[i] + 'PWDistances_0.pt', map_location=device)
    DT = pt.load(pw_data_dirs[i] + 'PWDistances_t.pt', map_location=device)

    R0_arr = np.loadtxt(pw_data_dirs[i] + 'R0.txt')


    Npoints = D0.shape[0]
    Ndims   = D0.shape[1]

    Nfinpoints  = DT.shape[1]
    Nframes     = DT.shape[3]


    frame = 9
    # Dt = pt.clone(DT[frame,:,:,:])

    Dt = DT[frame,:,:,:]

    print(Dt.shape)



    # Define the interpolating function
    f_NN = NeuralNetwork( Nodes = np.asarray(nodes) ).to(device)


    if i > 0 and os.path.exists(out_isokann +  f"f_NN_chk_{i-1}.pt"):
        f_NN.load_state_dict(pt.load(out_isokann +  f"f_NN_chk_{i-1}.pt", map_location=device))

    # npX0 = D0.cpu().detach().numpy()



    # Apply the power method using the best hyperparameters
    train_LOSS[i], val_LOSS[i], best_loss[i], convergence[i] = power_method(D0,
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

    chi[i]  = f_NN(D0).cpu().detach().numpy()

    pt.save(f_NN.state_dict(),out_isokann +  f"f_NN_chk_{i}.pt")

    # 7) FREE big tensors before next replica
    del D0, DT, Dt
    gc.collect()
    if device.type == 'cuda':
        pt.cuda.empty_cache()


# In[ ]:


pt.save(f_NN.state_dict(), out_isokann  + 'f_NN.pt')
np.save(out_isokann + 'val_LOSS.npy', val_LOSS)
np.save(out_isokann + 'train_LOSS.npy', train_LOSS)
np.save(out_isokann + 'chi0.npy', chi)


# # Calculate propagated chi
# #chit = f_NN(Xtau).cpu().detach().numpy()
# #np.save(out_isokann + 'chit.npy', chit)


# In[ ]:


# print('Correlation with end-to-end distance:', np.corrcoef(R0_arr, chi)[0,1])

font = {'size'   : 8}
plt.rc('font', **font)
in2cm = 1/2.54  # centimeters in inches

fig, ax = plt.subplots(6, 3, figsize=(16*in2cm, 6*in2cm), facecolor='white')

for i in range(len(pw_data_dirs)-1):
    pos = ax[i,0].scatter( chi[i], R0[i], c = chi[i],  cmap = cm.bwr , s = 5 )

    ax[i,0].set_title('$\chi$-function')
    ax[i,0].set_xlim((0,1))
    #ax[0].set_ylim(0,8)
    ax[i,0].set_ylabel(r'$r_{ee}$ / nm')
    ax[i,0].set_xlabel(r'$\chi$')


    ax[i,1].plot(train_LOSS[i], label='train loss')
    ax[i,1].plot(val_LOSS[i], label='validation loss')
    ax[i,1].semilogy()
    ax[i,1].set_xlabel(r'Iterations $\times$ epochs')
    ax[i,1].set_title('Loss functions')
    ax[i,1].legend()

    ax[i,2].plot(convergence[i])
    ax[i,2].semilogy()
    ax[i,2].set_xlabel('Iteration')
    ax[i,2].set_title(r'$\chi$-Convergence')
    ax[i,2].set_ylim(0.1,10)

plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.2, hspace=0.7)
fig.savefig(out_isokann + "isokann.png", format='png', dpi=300, bbox_inches='tight')


# In[ ]:




