#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import sys
import os
import glob

# Add the project root
sys.path.append(os.path.abspath('../../'))

from src.useful_functions import *
from src.openmm.PWDs_module1 import generate_PWDistances_torch

# For reproducibility
np.random.seed(0)

# For matplotlib
font = {'size'   : 10}
plt.rc('font', **font)
in2cm = 1/2.54  # centimeters in inches

# Read directory paths
read_dirs_paths('dir_paths.txt', globals())


# In[ ]:


if not os.path.exists(out_isokann):
    # Create the directory
    os.makedirs(out_isokann )
    print(out_isokann , "created successfully!")
else:
    print(out_isokann , "already exists!")

print(" ")
print(" ")

generate_PWDistances_torch(
                        inp_dir  =  inp_dir ,
                        out_dir  =  out_trajectories2,
                        iso_dir  =  out_trajectories1 ,
                        pdbfile_solute    = '2CM2_4884_v4.pdb', 
                        pdbfile_water     = 'pdbfile_water.pdb', 
                        prmtopfile_solute = 'prmtopfile_solute.prmtop', 
                        file_traj_water   = 'trajectory_water_combined.dcd',
                        frames     = np.array([0,1,2,3,4,5,6,7,8,9]),
                        rel_coords = np.array([[0,70]]),
                        BB=True
                        )



# In[ ]:




