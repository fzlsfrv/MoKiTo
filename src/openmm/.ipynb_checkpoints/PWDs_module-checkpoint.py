import MDAnalysis as mda
import os
import mdtraj as md
import numpy as np
import sympy as sp
import torch as pt
from tqdm import tqdm
import scipy
import itertools
import matplotlib.pyplot as plt
import glob
from openmm import *
from openmm.app import *
from openmm.unit import *
from scipy.spatial.distance import pdist



print(" ")
device = pt.device("cuda" if pt.cuda.is_available() else "cpu")


#pt.cuda.empty_cache()

def generate_pairs(N):
    #return np.c_[np.array(np.meshgrid(np.arange(N), np.arange(N))).T.reshape(-1,2)]
    t = np.arange(0,N,1)
    return np.array(list(set(itertools.combinations(t, 2))))

def generate_PWDistances_torch(
                        inp_dir  =  'input/',
                        out_dir  =  'output/',
                        iso_dir  =  'ISOKANN_files/',
                        pdbfile_solute    = 'pdbfile_no_water.pdb', 
                        pdbfile_water     = 'pdbfile_water.pdb', 
                        prmtopfile_solute = "prmtopfile_no_water.prmtop", 
                        file_traj_water   = "trajectory_water.dcd",
                        file_traj_solute  = "trajectory.dcd",
                        frames     = np.array([0,1,2,3,4,5,6,7,8,9]),
                        rel_coords = np.array([[0, 70]]),
                        periodic   = False,
                        BB=False):
    
    if BB==True:
        print(" ")
        print("This will create torch files containing PW distances using only BACKBONE atoms")
        print(" ")
    else:
        print(" ")
        print("This will create torch files containing PW distances using ALL atoms")
        print(" ")
    
    # Starting points (number of files in input/initial_states)
    traj      = md.load(out_dir + file_traj_water, top = inp_dir + pdbfile_water)   

    Npoints  = traj.n_frames
   
    print("Number of initial states:", Npoints)
    
    _, _, files = next(os.walk(out_dir + 'final_states/'))
    Nfinpoints = int(( len(files) - 1 ) / Npoints)
    print("Number of final states:", Nfinpoints)
    
    #from MDAnalysis.analysis.bat import BAT
    #R = BAT(traj)
    
    
    pdb = mda.Universe(inp_dir + pdbfile_solute)
    Natoms = pdb.atoms.n_atoms
    print("Number of atoms (no water):", Natoms)
    
    
    # Calculate relevant coordinate
    print("I am generating the relevant coordinate...")
    if len(rel_coords[0])==2:
        print("len(rel_coords)==2, then the relevant coordinate is a DISTANCE between 2 atoms")
        r        = np.squeeze(md.compute_distances(traj, rel_coords, periodic=periodic))
    elif len(rel_coords[0])==3:
        print("len(rel_coords)==3, then the relevant coordinate is an ANGLE between 3 atoms")
        r        = np.squeeze(md.compute_angles(traj, rel_coords, periodic=True))
    elif len(rel_coords[0])==4:
        print("len(rel_coords)==4, then the relevant coordinate is a DIHEDRAL between 4 atoms")
        r        = np.squeeze(md.compute_dihedrals(traj, rel_coords, periodic=True))
        
    np.savetxt(iso_dir + 'R0.txt', r)
    
    print(" ")
    
    if BB == True:
    
        u = mda.Universe(inp_dir + pdbfile_water)
        calphas = u.select_atoms("name CA")
        n_calpha = calphas.n_atoms
        bb_idx = calphas.indices
        
        lig = u.select_atoms("resname KB8 and not element H") 
        n_lig = lig.n_atoms
        kb8_idx = lig.indices
        sel_idx = np.unique(np.concatenate([bb_idx, kb8_idx]))
        
        
        
        #coor = pdb_numpy.Coor(inp_dir + pdbfile_water)
        #bb_idx = coor.get_index_select("protein and name CA")
        #kb8_idx = coor.get_index_select("resname KB8")        
        #sel_idx = np.unique(np.concatenate([bb_idx, kb8_idx]))

        
        pairs   =   generate_pairs(len(sel_idx))
        
        Ndims   =   len(pairs)
        print("Number of backbone atoms + ligand:", len(sel_idx))
        print("Number of pairwise distances between BACKBONE+LIGAND atoms:", Ndims)
    
        # Load initial states
        print("I am creating the tensor with the initial states...")
        #d0  =  md.compute_distances(traj.atom_slice(bb), pairs, periodic=False)

        d0 = np.zeros((Ndims, Nframes))
        for i in range(int(Nframes)):
            trj.trajectory[i]
            atom_coords = trj.atoms.positions
            d0[:, i] = self_distance_array(atom_coords, box = ref.dimensions)

        d0 = np.transpose(d0)
        D0  =  pt.tensor(d0, dtype=pt.float32, device=device)
        
    else:
        pairs   =   generate_pairs(Natoms)
        Ndims   =   len(pairs)
        print("Number of pairwise distances between ALL atoms:", Ndims)
            
        d0 = np.zeros((Ndims, Nframes))
        for i in range(int(Nframes)):
            trj.trajectory[i]
            atoms   =  trj.atoms[sel_idx]
            atom_coords = atoms.positions
            d0[:, i] = self_distance_array(atom_coords, box = ref.dimensions)

        d0 = np.transpose(d0)
        D0  =  pt.tensor(d0, dtype=pt.float32, device=device)
    
    
    pt.save(D0, iso_dir + 'PWDistances_0.pt')
    
    
    print('Shape of D0?')
    print('Npoints, Ndims')
    print(D0.shape)
    print(" ")    
    
    # Load one trajectory to calculate number of frames
    print("I am creating the tensor with the final states...")
    xt         =  mda.Universe(top = inp_dir + pdbfile_water,
                               out_dir + "final_states/xt_0_r0.dcd")
    Ntimesteps = xt.n_frames
    
    
    Ntimesteps = len(frames)
    
    Dt = pt.zeros((Ntimesteps, Npoints, Nfinpoints, Ndims), dtype = pt.float32, device=device)
    
    for i in tqdm(Nframes):
        
        for j in range(Nfinpoints):
            strj      =  mda.Universe(out_dir + "final_states/xt_" + str(i) + "_r" + str(j) + "_aligned.dcd", 
                                  top = inp_dir + pdbfile_water)
    
            
            for k in range(Ntimesteps):
                if BB == True:
                    strj.trajectory[k]
                    atoms   =  strj.atoms[sel_idx]
                    atom_coords = atoms.positions
                    dt         =  np.transpose(self_distance_array(atom_coords, box = ref.dimensions))
                else:
                    strj.trajectory[k]
                    atom_coords = strj.atoms.positions
                    dt         =  np.transpose(self_distance_array(atom_coords, box = ref.dimensions))
                    
                Dt[k,i,j,:]  =  pt.tensor(dt, dtype=pt.float32, device=device)

    
    pt.save(Dt, iso_dir + 'PWDistances_t.pt')
    
    print(" ")
    print('Shape of Dt?')
    print('Nframes, Npoints, Nfinpoints, Ndims')
    print(Dt.shape)
    
