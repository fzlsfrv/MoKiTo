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


best_hyperparams = {'Nepochs': 10, 'nodes': np.array([51040, 34026, 17013,     1]), 'learning_rate': 0.01, 'weight_decay': 0.005, 'batch_size': 100, 'patience': 5, 'act_fun': 'leakyrelu'}


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

    if os.path.exists(out_isokann +  "chk.pt"):
        f_NN.load_state_dict(pt.load(out_isokann +  "chk.pt", map_location=device))

    npX0 = D0.cpu().detach().numpy()



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

    pt.save(f_NN.state_dict(),out_isokann +  "chk.pt")

    # 7) FREE big tensors before next replica
    del D0, DT, Dt
    gc.collect()
    if device.type == 'cuda':
        pt.cuda.empty_cache()


# In[ ]:


# print('Correlation with end-to-end distance:', np.corrcoef(R0, chi)[0,1])

# font = {'size'   : 8}
# plt.rc('font', **font)
# in2cm = 1/2.54  # centimeters in inches

# fig, ax = plt.subplots(1, 3, figsize=(16*in2cm, 6*in2cm), facecolor='white')


# pos = ax[0].scatter( chi, R0, c = chi,  cmap = cm.bwr , s = 5 )

# ax[0].set_title('$\chi$-function')
# ax[0].set_xlim((0,1))
# #ax[0].set_ylim(0,8)
# ax[0].set_ylabel(r'$r_{ee}$ / nm')
# ax[0].set_xlabel(r'$\chi$')


# ax[1].plot(train_LOSS, label='train loss')
# ax[1].plot(val_LOSS, label='validation loss')
# ax[1].semilogy()
# ax[1].set_xlabel(r'Iterations $\times$ epochs')
# ax[1].set_title('Loss functions')
# ax[1].legend()

# ax[2].plot(convergence)
# ax[2].semilogy()
# ax[2].set_xlabel('Iteration')
# ax[2].set_title(r'$\chi$-Convergence')
# ax[2].set_ylim(0.1,10)

# plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.2, hspace=0.7)
# fig.savefig("output/figures/isokann.png", format='png', dpi=300, bbox_inches='tight')


# In[ ]:


pt.save(f_NN.state_dict(), out_isokann  + 'f_NN.pt')
np.savetxt(out_isokann  + 'val_LOSS.txt', val_LOSS)
np.savetxt(out_isokann  + 'train_LOSS.txt', train_LOSS)
np.savetxt(out_isokann + 'chi0.txt', chi)
np.savetxt(out_isokann + 'convergence.txt', convergence)
# # Calculate propagated chi
# #chit = f_NN(Xtau).cpu().detach().numpy()
# #np.save(out_isokann + 'chit.npy', chit)


# In[ ]:




