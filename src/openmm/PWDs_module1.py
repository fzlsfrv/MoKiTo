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
import pdb_numpy
import glob

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

    # traj_files = glob.glob(os.path.join(out_dir, file_traj_water))
    # traj_files = sorted(traj_files)
    # traj_list = [md.load_dcd(f, top=inp_dir + pdbfile_water) for f in traj_files]
    # traj_concat = md.join(traj_list)
    # traj_concat.save_dcd(os.path.join(iso_dir, 'trajectory_water_all.dcd'))
    # print("Combined trajectory saved as trajectory_water_all.dcd")

    
    traj     = md.load(iso_dir + 'trajectory_water_all.dcd', top = inp_dir + pdbfile_water)   

    Npoints  = traj.n_frames
   
    print("Number of initial states:", Npoints)

    #get the final state numbers
    final_dirs = glob.glob(os.path.join(out_dir, 'final_states'))
    files = []
    for d in final_dirs:
        _, _, f = next(os.walk(d))
        files.extend([os.path.join(d, x) for x in f])
    files = sorted(files)

        
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
        
        coor = pdb_numpy.Coor(inp_dir + pdbfile_water)
        bb_idx = coor.get_index_select("protein and name CA")
        kb8_idx = coor.get_index_select("resname KB8 and not name H*")        
        sel_idx = np.unique(np.concatenate([bb_idx, kb8_idx]))

        


        
        pairs   =   generate_pairs(len(sel_idx))
        
        Ndims   =   len(pairs)
        print("Number of backbone atoms + ligand:", len(sel_idx))
        print("Number of pairwise distances between BACKBONE+LIGAND atoms:", Ndims)
    
        # Load initial states
        print("I am creating the tensor with the initial states...")
        #d0  =  md.compute_distances(traj.atom_slice(bb), pairs, periodic=False)

        d0 = np.zeros((Npoints, Ndims))
        for i in tqdm(range(10)):
            d0[i] = pdist(traj.atom_slice(sel_idx)[i].xyz[0,:,:])
            
        D0  =  pt.tensor(d0, dtype=pt.float32, device=device)
        
    else:
        pairs   =   generate_pairs(Natoms)
        Ndims   =   len(pairs)
        print("Number of pairwise distances between ALL atoms:", Ndims)
            
        d0 = np.zeros((Npoints, Ndims))
        for i in tqdm(range(10)):
            d0[i,:] = pdist(traj[i].xyz[0,0:Natoms,:])

        D0  =  pt.tensor(d0, dtype=pt.float32, device=device)
    
    
    pt.save(D0, iso_dir + 'PWDistances_0_1.pt')
    
    
    print('Shape of D0?')
    print('Npoints, Ndims')
    print(D0.shape)
    print(" ")    
    
    # Load one trajectory to calculate number of frames
    # print("I am creating the tensor with the final states...")
    # xt         =  md.load(out_dir + "final_states/xt_0_r0.dcd", 
    #                               top = inp_dir + pdbfile_water)
    # print("The shape of a file xt_i_rj.dcd is", xt.xyz.shape)
    # Ntimesteps = xt.n_frames
    
    
    Ntimesteps = len(frames)
    
    Dt = pt.zeros((Ntimesteps, Npoints, Nfinpoints, Ndims), dtype = pt.float32, device=device)
    
    for l in range(6):
        for i in tqdm(range(Npoints)):
            
            for j in range(Nfinpoints):
                xt      =  md.load(f'/scratch/htc/fsafarov/2cm2_simulation/md2/output_{l}/trajectories/openmm_files/' + "final_states/xt_" + str(i) + "_r" + str(j) + ".dcd", 
                                      top = inp_dir + pdbfile_water)
                for k in range(Ntimesteps):
                    frame = frames[k]
                    if BB == True:
                        #dt         =  md.compute_distances(xt.atom_slice(bb)[frame], pairs, periodic=False)
                        dt         =  pdist(xt.atom_slice(sel_idx)[frame].xyz[0,:,:])
                    else:
                        #dt         =  md.compute_distances(xt[frame], pairs, periodic=False)
                        dt         =  pdist(xt[frame].xyz[0,0:Natoms,:])
                        
                    Dt[k,i,j,:]  =  pt.tensor(dt, dtype=pt.float32, device=device)

    
    pt.save(Dt, iso_dir + 'PWDistances_t_1.pt')
    
    print(" ")
    print('Shape of Dt?')
    print('Nframes, Npoints, Nfinpoints, Ndims')
    print(Dt.shape)
    
