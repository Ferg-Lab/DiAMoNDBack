import sys, os
sys.path.append('../denoising_diffusion_pytorch')

import glob
import mdtraj as md
import numpy as np
import copy
import matplotlib.pyplot as plt
from pathlib import Path
import time

import torch
from torch import nn 
from torch.utils import data
import pickle as pkl
from tqdm.autonotebook import tqdm

from Bio.SVDSuperimposer import SVDSuperimposer
from numpy import array, dot, set_printoptions
from joblib import Parallel, delayed

from denoising_diffusion_pytorch_backmap_combined import Unet1D, GaussianDiffusion, Trainer, Dataset_traj, cycle, num_to_groups
device = torch.device("cuda")

import sidechainnet as scn
from sidechainnet.structure.build_info import NUM_COORDS_PER_RES, SC_BUILD_INFO
from sidechainnet.structure.structure import coord_generator
from sidechainnet.utils.sequence import ONE_TO_THREE_LETTER_MAP
THREE_TO_ONE_LETTER_MAP = {y: x for x, y in ONE_TO_THREE_LETTER_MAP.items()}

ATOM_MAP_14 = {}
for one_letter in ONE_TO_THREE_LETTER_MAP.keys():
    ATOM_MAP_14[one_letter] = ["N", "CA", "C", "O"] + list(
        SC_BUILD_INFO[ONE_TO_THREE_LETTER_MAP[one_letter]]["atom-names"])
    ATOM_MAP_14[one_letter].extend(["PAD"] * (14 - len(ATOM_MAP_14[one_letter])))
    
# code to n_mask (for alignment without relying on zero idxs)
alphabet = ['A', 'C', 'D', 'E', 'F', 'G','H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

n_masked_dict = {}
for a in ATOM_MAP_14:
    a_map = ATOM_MAP_14[a]
    m_cnt = 0
    for m in a_map:
        if m != 'PAD':
            m_cnt += 1
    n_masked_dict[a] = m_cnt
    
with open('../denoising_diffusion_pytorch/res_add_dict.pkl', 'rb') as handle:
    res_add_dict = pkl.load(handle)
    
def encode_seq(sequence):
    '''Encode AA to one-hot'''
    
    alphabet = ['A', 'C', 'D', 'E', 'F', 'G','H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    integer_encoded = [char_to_int[char] for char in sequence]
    onehot_encoded = list()

    for value in integer_encoded:
        letter = [0 for _ in range(len(alphabet))]
        letter[value] = 1
        onehot_encoded.append(letter)

    return np.array(onehot_encoded)

def sort_Ca(Ca_coords, Ca_idx, n_keep=10, exclude_neighbors=True):
    '''sort Ca distances from closest to farthest
       main_idx is the Ca of interest, other_idxs are non-adjacent Cas
       return n_frames x n_Ca 
       '''
    
    n_Ca = len(Ca_coords)
    dists = np.linalg.norm(Ca_coords - Ca_coords[Ca_idx], axis=-1) 

    # exclude Ca_main and two neighbors
    if exclude_neighbors:
        idxs_sorted = np.argsort(dists)[3:3+n_keep]
    else:
        idxs_sorted = np.argsort(dists)[1:1+n_keep]

    # return just the indices and the Ca xyz coordinaites
    return idxs_sorted, Ca_coords[idxs_sorted]


def sort_Ca_all_sc(all_coords, Ca_coords, Ca_idx, n_keep=10, com_shift=False, round2=False,
                   ter=[], ignore_ter=False):
    '''sort Ca distances from closest to farthest
       based on any residues decoded so far (assuming N-->C)
       '''
    
    # compare agaist Ca position of Ca shifted by the sidechain COM
    if com_shift:
        shift = np.array([0, 3.75, 1.04])
    else:
        shift = np.array([0, 0, 0])
    
    # get dists from all Cas first
    n_Ca = len(Ca_coords)
    dists = np.linalg.norm(Ca_coords - (Ca_coords[Ca_idx]+shift), axis=-1) 
    
    # for all previously decoded consider other atoms
    a_per_res = 14
    
    # if round2, get to see all residues
    if round2:
        Ca_idx_list = np.arange(n_Ca)
        
    # look at all previously decoded (N side) up to current idx (Ca_idx)
    else:
        Ca_idx_list = np.arange(Ca_idx)
       
    # don't allow to train on termini
    if ignore_ter:
        Ca_idx_list = np.array([i for i in Ca_idx_list if i not in ter])
    
    for Ca_s_idx in Ca_idx_list:

        # don't include zero_idxs
        idx_list = np.array([i for i in range(Ca_s_idx*a_per_res, (Ca_s_idx+1)*a_per_res) if not np.all(all_coords[i]==0)])
        
        all_res_dists = np.linalg.norm(all_coords[idx_list] - (Ca_coords[Ca_idx]+shift), axis=-1) 
        
        # replace with minimum distance for that residue
        dists[Ca_s_idx] = np.min(all_res_dists)
    
    # exclude Ca_main
    idxs_sorted = np.argsort(dists)[1:1+n_keep]

    # return just the indices and the Ca xyz coordinates
    return idxs_sorted, Ca_coords[idxs_sorted]

# residue builder information located here
# https://github.com/jonathanking/sidechainnet/blob/ec5b5b6d88f105510b6427a4489ee2471c729290/sidechainnet/structure/StructureBuilder.py#L256

def prepare_pdbs(pdb_name_list, n_samples=1, data_type='aa', top=None, stride=None):
    '''Inference for a given model and list of PDB'''
    
    n_frames = 0
    end_idx_list = [0]
    all_seqs, all_crds, all_ters, n_res_list = [], [], [], []

    for pdb_name in pdb_name_list*n_samples:
        
        if top is not None:
            trj = md.load(pdb_name, top=top, stride=stride)
        else:
            trj = md.load(pdb_name, stride=stride)
            
        # remove all hydrogens
        heavy_idxs = trj.top.select("mass > 1.1")
        if len(heavy_idxs) != trj.n_atoms and len(heavy_idxs) >  0 :
            trj = trj.atom_slice(heavy_idxs)
        
        # resort residues to account for effects of missing hydrogens
        n_res = trj.top.n_residues
        print(pdb_name.split('/')[-1], trj.xyz.shape, trj.top.n_residues, trj.top.n_chains)

        # passing in aa or cg data -- should make this produce a unified output
        if data_type=='aa':
            seqs, crds, ters = mdtrj_to_scnet(trj)
        elif data_type == 'cg':
            seqs, crds, ters = cg_to_scnet(trj)

        all_seqs += seqs
        all_crds += crds
        all_ters += ters

        n_frames = n_frames + len(trj)
        end_idx_list.append(n_frames)

        # recover n_res - ter for each seq
        n_res_list += [n_res-len(ters[0])]*len(trj)
    
    # get sampling indices
    n_sample_list = np.concatenate([[n]*len(pdb_name_list) for n in range(n_samples)])
    n_res_list = np.array(n_res_list)
        
    return (all_seqs, all_crds, all_ters, end_idx_list, n_sample_list, n_res_list)
        

def gen_aligned_xyzs(seq, crd, ter, cond_type='Closest-Cas', n_cond_res=10, 
                     exclude_neighbors=False, single_resid=None, round2=False, short='zero'):
    '''Return a list of xyz for a given sequence and coordinate set
       Specify single_resid for iterative decoding where only one residue at time is aligned'''

    # taken from average values in ScNet data
    b_avg, a_avg = 10*0.381, np.pi*(106.0/2)/180
    
    # ensures COM at 0,0,0
    if 'acenter' in cond_type:
        x, y = b_avg*np.sin(a_avg), b_avg*np.cos(a_avg)   # new
        y_shift = 2*b_avg*np.cos(a_avg) / 3
        Ca_triplet_ref = np.array([[-x, -y+y_shift, 0], 
                          [0, y_shift, 0],
                          [x, -y+y_shift, 0]])
        
    else:
        x, y = b_avg*np.cos(a_avg), b_avg*np.sin(a_avg)
        Ca_triplet_ref = np.array([[-x, -y, 0], 
                                  [0, 0, 0],
                                  [x, -y, 0]])
        
    # whether or not to condition on terminal sidechains
    if 'ignT' in cond_type:
        ignore_ter = True
    else:
        ignore_ter = False
        
    # whether to shift the interaction reference point by sc com
    if '-com-' in cond_type:
        com_shift = True
    else:
        com_shift = False

    n_res = len(seq)
    a_per_res = 14 # this is always 14
    xyz_list = []
    
    # exclude all terminal indices
    Ca_idx_list = [i for i in range(n_res) if i not in ter]
    
    # for iterative decoding only align one residue
    if single_resid is not None:
        Ca_idx_list = [Ca_idx_list[single_resid]]
    
    for Ca_idx in Ca_idx_list:

        Ca_triplet_idxs = (np.array([-1,0,1])+Ca_idx)*a_per_res + 1
        Ca_triplet_xyz = crd[Ca_triplet_idxs]

        # superpose on fixed reference
        sup = SVDSuperimposer()
        sup.set(Ca_triplet_ref, Ca_triplet_xyz)
        sup.run()
        rms = sup.get_rms()
        rot, tran = sup.get_rotran()

        # translate xyz only then full coords
        xyz_on_ref = dot(Ca_triplet_xyz, rot) + tran
        crd_on_ref = dot(crd, rot) + tran

        # make sure to keep zero values after superpose
        zero_idxs = np.where(np.all(crd==0, axis=1))[0]   #np.where(np.sum(crd, axis=1)==0)[0]
        crd_on_ref[zero_idxs] = crd[zero_idxs]

        # extract Ca-1 backbone section
        bb1_idxs = np.arange((Ca_idx-1)*a_per_res+1, (Ca_idx-1)*a_per_res+4)
        sc_idxs = np.arange((Ca_idx)*a_per_res, (Ca_idx+1)*a_per_res)
        bb2_idxs = np.arange((Ca_idx+1)*a_per_res, (Ca_idx+1)*a_per_res+2)

        # put all aligned pieces together
        res_idxs = np.concatenate([bb1_idxs, sc_idxs, bb2_idxs])
        xyz = crd_on_ref[res_idxs]
        
        # All Ca values
        all_Ca_idxs = np.arange(len(seq))*a_per_res + 1
        all_Ca_coords = crd_on_ref[all_Ca_idxs]
        
        # append xyz with whatever xyz 
        if 'Closest-Cas' in cond_type:

            # idxs sorted and add to xyz
            Ca_sorted_idxs, all_Ca_sorted = sort_Ca(all_Ca_coords, Ca_idx, 
                                    n_keep=n_cond_res, exclude_neighbors=exclude_neighbors)
            xyz = np.concatenate([xyz, all_Ca_sorted])
            
            # 20-dim ohe
            ohe = encode_seq(seq[Ca_idx])[0]
            xyz = np.concatenate([xyz.flatten(), ohe])
           
        elif 'Closest-scs-N2C' in cond_type:
            
            # idxs sorted either by Ca distance or dist to nearest sidechain
            if 'resdist' in cond_type:
                
                Ca_sorted_idxs, all_Ca_sorted = sort_Ca_all_sc(crd_on_ref, all_Ca_coords, 
                                                           Ca_idx, n_keep=n_cond_res, com_shift=com_shift, 
                                                               round2=round2, ter=ter, ignore_ter=ignore_ter)
                
            else:
                Ca_sorted_idxs, all_Ca_sorted = sort_Ca(all_Ca_coords, Ca_idx, 
                                            n_keep=n_cond_res, exclude_neighbors=exclude_neighbors)

            block_size = a_per_res - 3 # for 10 sidechain + 1 Ca 
            xyz_cond = np.zeros((n_cond_res*block_size, 3))
          
            # fill Ca conditioning block
            for i, Ca_s_idx in enumerate(Ca_sorted_idxs):
                
                # update Ca positions for all closest Cas
                Ca_crd_idx = Ca_s_idx*a_per_res + 1
                xyz_cond[block_size*i] = crd_on_ref[Ca_crd_idx]
                
                # collect all SC info if already decoded
                # or if round2, always collect all info
                if Ca_s_idx < Ca_idx or round2: # or Ca_s_idx in ter:
                    sc_idx_first, sc_idx_last = Ca_s_idx*a_per_res + 4, (Ca_s_idx+1)*a_per_res
                    xyz_cond[block_size*i+1: block_size*(i+1)] = crd_on_ref[sc_idx_first: sc_idx_last]
                    
                # fill in blank sidechain blocks with Ca values, otherwise leave them as zeros
                elif 'fill-Cas' in cond_type:
                    
                    for sc_pos in range(block_size-1):
                        cond_idx = block_size*i + sc_pos + 1
                        crd_idx = Ca_s_idx*a_per_res + sc_pos + 4
                        
                        # break if hit all zeros and keep remainin entries at zero
                        if np.sum(crd[crd_idx]) != 0:
                            xyz_cond[cond_idx] = crd_on_ref[Ca_crd_idx]
                        else:  break
                            
            # if protein is shorter than just repeat last values n_cond_res - N
            last_i = len(Ca_sorted_idxs) - 1
            if last_i+1 < n_cond_res:
                
                if short=='all':
                    last_cond = xyz_cond[block_size*last_i: block_size*(last_i+1)] 
                    for i in range(last_i+1, n_cond_res):
                        xyz_cond[block_size*i: block_size*(i+1)] = last_cond
                        
                elif short=='Ca':
                    last_Ca = xyz_cond[block_size*last_i] 
                    for i in range(last_i+1, n_cond_res):
                        xyz_cond[block_size*i] = last_Ca
                        
                else:
                    pass

            # append to residue input
            xyz = np.concatenate([xyz, xyz_cond])
            
            # 20-dim ohe
            ohe = encode_seq(seq[Ca_idx])[0]
            xyz = np.concatenate([xyz.flatten(), ohe])
            
        xyz_list.append(xyz)
    return xyz_list


def load_trainer(data_name, model_name, folder_name = './per_res_data',  last_milestone=None, 
                 model_train_steps = 2_000_000, dtime = 50, init_dim = 32, 
                 dim_mults = (1, 2, 4, 8), train_batch_size = 128, resmask=False):
  
    train_name = f'train_{data_name}' 
    cond_name = f'test_{data_name}_cond' 
    shared_name = f'test_{data_name}_shared_idxs'

    channels = 1
    loss = 'l1'
    beta = 'cosine'

    model = Unet1D(dim = init_dim, dim_mults = dim_mults, channels=1) 

    # load shared idxs for mask
    shared_idxs = np.load(f'{folder_name}/{shared_name}.npy')
    op_number = np.load(f'{folder_name}/{train_name}_traj.npy').shape[-1]

    model = nn.DataParallel(model)
    model.to(device)

    diffusion = GaussianDiffusion(
        model,                          # U-net model
        timesteps = dtime,              # number of diffusion steps
        loss_type = loss,               # L1 or L2
        beta_schedule = beta,
        unmask_list = shared_idxs,
    ).to(device) 

    #set training parameters
    trainer = Trainer(
        diffusion,                                   # diffusion model
        folder = folder_name,                        # folder of trajectories
        system = train_name,         
        train_batch_size = 128,                      # training batch size
        train_lr = 1e-5,                             # learning rate
        train_num_steps = model_train_steps,         # total training steps
        gradient_accumulate_every = 1,               # gradient accumulation steps
        ema_decay = 0.995,                           # exponential moving average decay
        op_number = op_number,
        fp16 = False,
        save_name = model_name,      # turn on mixed precision training with apex
    )

    # find last milestone
    trainer.RESULTS_FOLDER = Path(f'../{trainer.RESULTS_FOLDER}')
    
    model_list = sorted(glob.glob(f'{trainer.RESULTS_FOLDER}/model-*.pt'))
    print(trainer.RESULTS_FOLDER, len(model_list))
    
    if last_milestone is None:
        last_milestone = max([int(m.split('/')[-1].replace('.pt', '').replace('model-', '')) for m in model_list])
        print(f'Milestone = {last_milestone}')
    
    # start load last 
    trainer.load(last_milestone)

    return trainer, last_milestone


def run_inference(trainer, npy_data=None, shared_idxs=None, data_name=None, 
                  last_milestone=1, folder_name = 'per_res_data', 
                  batch_size = 5000, batches=None, save_to_file=False, save_diffusion_progress=False):
    '''Performs inference solely based on what is in the conditioning file
       (Not iteratively)'''
        
    # only if loading from folder
    if type(data_name)==str:
       
        # find all related files from data_name
        cond_name = f'test_{data_name}_cond' 
        shared_name = f'test_{data_name}_shared_idxs'
        shared_idxs = np.load(f'{folder_name}/{shared_name}.npy')
    
        # prepare a dataloader to give samples from the conditional part of the distribution
        sample_ds = Dataset_traj(folder_name, cond_name, shared_idxs=shared_idxs)
        
    # otherwise pass in npy directly as sample_ds
    else:
        sample_ds = Dataset_traj(npy_data=npy_data, shared_idxs=shared_idxs)
        
    sample_ds.max_data = trainer.ds.max_data
    sample_ds.min_data = trainer.ds.min_data    #To ensure that the sample data is scaled in the same way as the training data
    
    # total number of samples
    num_sample = sample_ds.data.shape[0] 
    
    if batches is None:
        batches = num_to_groups(num_sample, batch_size)

    # both shuffle and pin_memory og set to true
    sample_dl = cycle(data.DataLoader(sample_ds, batch_size = batch_size, shuffle=False, pin_memory=False)) 
    
    # account for differnt ddpm.py version -- unify this
    if save_diffusion_progress:
        all_ops_list = list(map(lambda n: trainer.ema_model.sample(
            trainer.op_number, batch_size=n, samples = next(sample_dl).cuda()[:n, :], 
            save_diffusion_progress=save_diffusion_progress), batches))
        
    else:
        all_ops_list = list(map(lambda n: trainer.ema_model.sample(
            trainer.op_number, batch_size=n, samples = next(sample_dl).cuda()[:n, :]), batches))

    all_ops = torch.cat(all_ops_list, dim=0).cpu()
    all_ops = trainer.rescale_sample_back(all_ops)

    if save_to_file:
        #np.save(str(trainer.RESULTS_FOLDER / f'samples_final-m{last_milestone}'), all_ops.numpy())
        np.save(str(trainer.RESULTS_FOLDER / f'test'), all_ops.numpy())
    
    return all_ops.numpy().squeeze(axis=1)

def align_to_backbone(seq, crd, ter, xyz_list, single_resid=None):
    '''Return crd format for an input of generated xyz coordinates by superposing on Ca backbone
       For single_resid, only superpose one reside and return crd_gen array'''

    n_res = len(seq)
    a_per_res = 14 # this is always 14
    pred_size = 19 # asuming Ca triplet with overlaps
    xyz_triplet_idxs = np.array([0, 4, pred_size-1]) # should always be consistnet for Cas
    
    # extract all backbone info from crd and copy
    all_Ca_idxs = np.arange(len(seq))*a_per_res + 1
    all_Ca_bb = crd[all_Ca_idxs] 
    crd_gen = copy.deepcopy(crd)
    
    # exclude all terminal indices and find corresponding Ca idx
    Ca_idx_list = [i for i in range(n_res) if i not in ter]
  
    # for iterative decoding only align one residue
    if single_resid is not None:
        Ca_idx_list = [Ca_idx_list[single_resid]]

    for xyz_idx, Ca_idx in enumerate(Ca_idx_list):
        
        # coordinate to align against
        Ca_triplet_bb = all_Ca_bb[np.array([-1,0,1])+Ca_idx]
            
        xyz_gen = xyz_list[xyz_idx][:pred_size*3].reshape(pred_size, 3)
        Ca_triplet_gen = xyz_gen[xyz_triplet_idxs]

        # superpose on fixed reference
        sup = SVDSuperimposer()
        sup.set(Ca_triplet_bb, Ca_triplet_gen)
        sup.run()
        rms = sup.get_rms()
        rot, tran = sup.get_rotran()

        # superpose xyz only (just to verify) 
        trip_on_ref = dot(Ca_triplet_gen, rot) + tran
        
        # superpose all coordinates
        gen_on_ref = dot(xyz_gen, rot) + tran

        # extract Ca-1 backbone section -- note this ordering is different from CLN
        bb1_idxs = np.arange((Ca_idx-1)*a_per_res+1, (Ca_idx-1)*a_per_res+4)
        sc_idxs = np.arange((Ca_idx)*a_per_res, (Ca_idx+1)*a_per_res)
        bb2_idxs = np.arange((Ca_idx+1)*a_per_res, (Ca_idx+1)*a_per_res+2)
        replace_idxs = np.concatenate([bb1_idxs, sc_idxs, bb2_idxs])
     
        # overwrite residues idxs
        crd_gen[replace_idxs] = gen_on_ref
        
        # make sure to keep zero values after superpose -- but don't include zeros within residue
        zero_idxs = np.where(np.all(crd==0, axis=1))[0]
        
        # unmask residue values for cg trajs (overides zeros)
        n_unmskd = n_masked_dict[seq[Ca_idx]]
        unmskd_list = [n for n in range(Ca_idx*a_per_res, Ca_idx*a_per_res + n_unmskd)]

        # also unmask any backbone predictions for adjacent residues -- only matters for CG
        unmskd_list +=  [(Ca_idx-1)*a_per_res+2, (Ca_idx-1)*a_per_res+3, (Ca_idx+1)*a_per_res]

        # remove all unmasked from zero lsit
        zero_idxs = list( set(zero_idxs).difference(set(unmskd_list)) )

        # reset to zero
        crd_gen[zero_idxs] = crd[zero_idxs]
        
    return crd_gen

def mdtrj_to_scnet(trj):
    '''Convert mdtraj to list of seqs and crds in sidechainnet format'''

    # just one frame one at a time for now:
    all_seqs, all_crds, all_ters = [], [], []
    
    # get terminal indices (to exclude for aligment later on)
    ter_list = []
    ter_atoms = 0
    for chain in trj.top.chains:
        len_chain = len(list(chain.residues))
        ter_list += [ter_atoms, ter_atoms+len_chain-1]
        ter_atoms += len_chain
    
    # make a dictionary of resid:atomtype:atom index -- exclude termini
    res_atype_dict = {}
    
    for j, a in enumerate(trj.top.atoms):
        
        # out of order atoms in pdb can break this -- process through table sort first
        resid = f'{a.residue.name}{a.residue.index}'
        name = a.name

        # make a dict for each unique resid
        if resid not in res_atype_dict:
            res_atype_dict[resid] = {name:j}       
        else:
            res_atype_dict[resid][name] = j
          
    for frame in range(len(trj)):

        # load in atoms one by one such that they match 
        crd, seq = [], []
        for i, res in enumerate(res_atype_dict.keys()):
            
            one_letter = THREE_TO_ONE_LETTER_MAP[res[:3]]
            crd_map = ATOM_MAP_14[one_letter]

            # collect indices for each map
            crd_idx_list = []
            for c in crd_map:
                if c=='PAD': break
                crd_idx_list.append(res_atype_dict[res][c])
            crd_idx_list = np.array(crd_idx_list)

            # collect xyz values from full traj -- and keep pad values as zero
            crd_res = np.zeros((14, 3))
            crd_res[:len(crd_idx_list)] = trj.xyz[frame, crd_idx_list]

            # collect across all residues
            seq.append(one_letter)
            crd.append(crd_res)

        # join to match sidechnet format
        seq = ''.join(seq)
        crd = np.concatenate(crd)*10

        # keep list of all crds
        all_seqs.append(seq)
        all_crds.append(crd)
        all_ters.append(ter_list)
        
    return all_seqs, all_crds, all_ters

def cg_to_scnet(trj):
    '''Convert mdtraj of CG protein to list of seqs and crds in sidechainnet format'''

    # just one frame one at a time for now:
    all_seqs, all_crds, all_ters = [], [], []
    
    # get terminal indices (to exclude for aligment later on)
    ter_list = []
    ter_atoms = 0
    for chain in trj.top.chains:
        len_chain = len(list(chain.residues))
        ter_list += [ter_atoms, ter_atoms+len_chain-1]
        ter_atoms += len_chain
    
    for frame in range(len(trj)):

        # load in atoms one by one such that they match 
        crd, seq = [], []
        for i, res in enumerate(trj.top.residues):
            one_letter = THREE_TO_ONE_LETTER_MAP[res.name]
            crd_map = ATOM_MAP_14[one_letter]
            
            # all zero except for the Ca (1) idx
            crd_res = np.zeros((14, 3))
            crd_res[1] = trj.xyz[frame, i]

            # collect across all residues
            seq.append(one_letter)
            crd.append(crd_res)

        # join to match sidechnet format
        seq = ''.join(seq)
        crd = np.concatenate(crd)*10

        # keep list of all crds
        all_seqs.append(seq)
        all_crds.append(crd)
        all_ters.append(ter_list)
        
    return all_seqs, all_crds, all_ters

def check_Ca_bonds(xyz_list, Ca_bond_range=(2.7, 4.1)):
    '''Looks for any unusually long/short Ca bonds outside the indicated range'''
              
    xyz_list = np.array(xyz_list)[:, :19*3].reshape(-1, 19, 3)
              
    ba = xyz_list[:, 0] - xyz_list[:, 4]
    bc = xyz_list[:, 18] - xyz_list[:, 4]   
              
    d1_d2 = np.concatenate([np.linalg.norm(ba, axis=1), np.linalg.norm(bc, axis=1)])
              
    if np.min(d1_d2) < Ca_bond_range[0] or np.max(d1_d2) > Ca_bond_range[1]:
        return False
    else:
        return True
    
def check_res_masks(xyz_list):
    '''Ensure that the correct number of atoms are included for each residue'''

    n_bad_res = 0
    ohe_to_nmask_dict = {14: 7, 6: 6, 8: 5, 2: 4, 3: 5, 15: 2, 16: 3, 11: 4, 13: 5, 1: 2, 5: 0, 12: 3, 0: 1, 17: 3, 7: 4, 9: 4, 10: 4, 4: 7, 19: 8, 18: 10}
      
    ohe_list = np.array(xyz_list)[:, -20:]
    sc_list = np.array(xyz_list)[:, 7*3:17*3]
        
    for ohe, sc in zip(ohe_list, sc_list):
    
        ohe_idx = int(np.where(ohe>0)[0])
        nmask = ohe_to_nmask_dict[ohe_idx]

        sc_block = sc.reshape(-1, 3)
        fill_block = sc_block[:nmask]

        if len(np.where(np.all(fill_block==0, axis=1))[0]) > 0:
            n_bad_res += 1
            
    if n_bad_res > 0:
        return False
    else:
        return True

def save_as_test(all_seqs, all_crds, all_ters, save_dir, prefix, save_type='test',
                 cond_type='Closest-Cas', n_cond_res=10, exclude_neighbors=False, single_resid=None, 
                 round2=False, short='zero', remove_zero_res=False, remove_bad_bonds=False,
                 augment_TER=1):
    '''Save all samples as an aligned and featurized list'''
    
    #Parallel(n_jobs=8)(delayed(find_min_dist)(Ca_s_idx) for Ca_s_idx in range(Ca_idx))
    print('Starting alignment')
    
    test_xyzs = []
    bond_rejects = []
    res_rejects = []
              
    for i, (seq, crd, ter) in tqdm(enumerate(zip(all_seqs, all_crds, all_ters)), total=len(all_seqs)):
        
        # align and add to test list -- should pass in arbitrary condition funciton here
        xyz_list = gen_aligned_xyzs(seq, crd, ter, cond_type=cond_type, n_cond_res=n_cond_res, 
                                    exclude_neighbors=exclude_neighbors, single_resid=single_resid,
                                    round2=round2, short=short)
    
        # check xyz_list for correct bonds and residues masking, otherwise append bad idx (no not include termini in check)
        passed_bonds, passed_res = True, True
        if remove_bad_bonds:
            if not check_Ca_bonds(xyz_list):
                passed_bonds = False
                bond_rejects.append(i)
                
        # include terminal residues
        if 'TER' in cond_type and save_type=='train':
            xyz_TER = gen_aligned_xyzs_terminal(seq, crd, ter, cond_type=cond_type, 
                                                  n_cond_res=n_cond_res, 
                                                  exclude_neighbors=exclude_neighbors, 
                                                  single_resid=single_resid, short=short)
            
            # augment training data with additional terminal residues
            xyz_list += xyz_TER * augment_TER

        # look for any residues with missing atoms
        if remove_zero_res:
            if not check_res_masks(xyz_list):
                passed_res = False
                res_rejects.append(i)
     
        # only accept sequence xyzs if both tests are passed
        if passed_bonds and passed_res:
            test_xyzs += xyz_list
        
        # check progress
        if i % 1000==0 and i>0: 
            Ca_vals = np.array(test_xyzs)[:, np.array([1, 13, 55])]
            min_vals = np.round(np.min(Ca_vals, axis=0), 2)
            max_vals = np.round(np.max(Ca_vals, axis=0), 2)
            
            print('\n', i, '/', len(all_seqs), np.shape(test_xyzs))
            print(len(bond_rejects), len(res_rejects))

    test_xyzs = np.array(test_xyzs)
    
    # this should be generally applicable to any conditioning on a 19-met input
    mask_idxs = list(np.concatenate([np.arange(i*3,i*3+3) for i in [0,4,18]])) + list(np.arange(19*3, len(xyz_list[0])))
    mask_idxs = np.array(mask_idxs)
    
    # don't save for inference, just output files directly
    if save_type!='inf':

        np.save(f'{save_dir}/{save_type}_{prefix}_traj.npy', test_xyzs)
        np.save(f'{save_dir}/test_{prefix}_shared_idxs.npy', mask_idxs)

        if save_type=='test':
            np.save(f'{save_dir}/{save_type}_{prefix}_cond_traj.npy', test_xyzs[:, mask_idxs])
    
    return test_xyzs, mask_idxs, bond_rejects, res_rejects
    
    
def get_crd_gen(all_seqs, all_crds, all_ters, gen_xyzs):
    '''piece gen_xyzs back together on Ca backbone -- assumes you're not touching the termini
       Need a list of all seqs and reference coords + generated xyzs'''
    
    xyz_idx = 0
    crd_gen_list = []
    for seq, crd, ter in zip(all_seqs, all_crds, all_ters):

        # align xyz back on to the backbone
        len_nonter = len(seq) - len(ter)
        seq_xyzs = gen_xyzs[xyz_idx:xyz_idx+len_nonter]
        crd_gen = align_to_backbone(seq, crd, ter, seq_xyzs)

        # check ohe match between the real an gne
        Ca_idx_list = np.array([i for i in range(len(seq)) if i not in ter])
        ohe_real = encode_seq(''.join([seq[i] for i in Ca_idx_list])) 
        ohe_gen = seq_xyzs[:, -20:]
        #print('Zero if OHEs match: ', np.sum(ohe_real-ohe_gen))

        crd_gen_list.append(crd_gen)
        xyz_idx += len_nonter

        # break if run out of samples
        if len(gen_xyzs) - xyz_idx < 1:
            break

    return crd_gen_list


def crd_to_mdtrj(all_seqs, crd_ref_list, crd_gen_list, end_idx_list, 
                 name_list, n_sample_list=None, save_dir='./iter_gens/test/'):
    '''Save reference and generated trajectories as pdbs'''

    # convert to mdtrj format
    ref_pdb_list, gen_pdb_list = [], []
    try: os.mkdir(save_dir)
    except: pass
        
    # create full pdbs from individual frames
    for i, (s, c, c_gen) in enumerate(zip(all_seqs, crd_ref_list, crd_gen_list)):
        
        # slightly perturb any that sums to equal exactly zero, but that do not all equal zero (scnet pdb bug)
        zref_idx = np.where( (np.sum(c, axis=1)==0) & (np.all(c, axis=1)!=0) )[0]
        c[zref_idx] += 0.000001
        
        zgen_idx = np.where( (np.sum(c_gen, axis=1)==0) & (np.all(c_gen, axis=1)!=0) )[0]
        c_gen[zgen_idx] += 0.000001

        sb_ref = scn.StructureBuilder(s, crd=c)
        sb_ref.to_pdb('ref_temp.pdb')

        sb_gen = scn.StructureBuilder(s, crd=c_gen)
        sb_gen.to_pdb('gen_temp.pdb')

        trj_ref = md.load('ref_temp.pdb')
        trj_gen = md.load('gen_temp.pdb')
    
        ref_pdb_list.append(trj_ref)
        gen_pdb_list.append(trj_gen)
        
    if n_sample_list is None:
        n_sample_list = [0]*len(name_list)

    # regroup frames from shared PEDs together
    for idx, (start, end, name, n_sample) in enumerate(
        zip(end_idx_list[:-1], end_idx_list[1:], name_list, n_sample_list)):

        # ensure all tests share the same topology
        trj_ref = md.join(ref_pdb_list[start:end])
        trj_gen = md.join(gen_pdb_list[start:end])
        print(trj_ref.xyz.shape, trj_gen.xyz.shape)
        
        trj_ref.save(f'{save_dir}/{name}_ref.pdb')
        trj_gen.save(f'{save_dir}/{name}_gen_{n_sample}.pdb')


####  For terminal trajs only ##### 

def gen_aligned_xyzs_terminal(seq, crd, ter, cond_type='Closest-Cas', n_cond_res=10, 
                             exclude_neighbors=False, single_resid=None, short='zero'):
    
    '''Return a list of xyz for a given sequence and coordinate set
       Specify single_resid for iterative decoding where only one residue at time is aligned'''
    
    # termini will always be exposed to all deocded residues
    round2 = True
    
    # whether to shift the interaction reference point by sc com
    if '-com-' in cond_type:
        com_shift = True
    else:
        com_shift = False
        
    # whether or not to condition on terminal sidechains
    if 'ignT' in cond_type:
        ignore_ter = True
    else:
        ignore_ter = False

    n_res = len(seq)
    a_per_res = 14 # this is always 14 based on scnet block
    xyz_list = []
    
    # decode one terminal at a time, or all termini at once
    # NC == 0 indicates N-terminal, NC == 1 indicates C-terminal
    if single_resid is None:
        ter_idx_list = sorted(ter)
        NC_list = [0, 1] * (len(ter) // 2)
    else:
        ter_idx_list = [sorted(ter)[single_resid]]
        NC_list = [single_resid%2]
        
    # identify each Ca_idx 
    for NC, Ca_idx in zip(NC_list, ter_idx_list):

        # for N terminus
        if NC==0:

            # Ca, C, Ca+1
            backbone_idxs = np.array([1,2,a_per_res+1])

            # taken frome average idxs across all scnet config (acenter)
            backbone_ref = np.array([[-1.00574234e-04,  1.49186248e+00, -8.33883672e-19],
                                     [ 1.22161215e+00,  6.95844205e-01,  3.81352596e-02],
                                     [ 3.02203985e+00, -7.45969134e-01,  3.62298962e-19]])

        # for C-terminus
        else:

            # Ca-1, C-1, Ca
            backbone_idxs = np.array([-a_per_res+1,-a_per_res+2,1])

            # taken frome average idxs across all scnet config (acenter)
            backbone_ref = np.array([[-3.02193927e+00, -7.45893346e-01,  2.40498757e-19],
                                     [-1.83997494e+00,  4.92908752e-02, -4.09291509e-01],
                                     [-1.00574234e-04,  1.49186248e+00, -8.33883672e-19]])

        backbone_idxs = Ca_idx*a_per_res + backbone_idxs
        backbone_xyz = crd[backbone_idxs]

        # superpose on fixed reference
        sup = SVDSuperimposer()
        sup.set(backbone_ref, backbone_xyz)
        sup.run()
        rms = sup.get_rms()
        rot, tran = sup.get_rotran()

        # translate xyz for full coords
        crd_on_ref = dot(crd, rot) + tran
        xyz_on_ref = dot(backbone_xyz, rot) + tran

        # make sure to keep zero values after superpose
        zero_idxs = np.where(np.all(crd==0, axis=1))[0]   #np.where(np.sum(crd, axis=1)==0)[0]
        crd_on_ref[zero_idxs] = crd[zero_idxs]

        # prepare empty array and fill Ca_i position
        xyz = np.zeros((19, 3))

        # both termini share all central residues
        sc_idxs = np.arange((Ca_idx)*a_per_res, (Ca_idx+1)*a_per_res)
        xyz[3:17] = crd_on_ref[sc_idxs]

        # for N terminus
        if NC==0:
            Ca_plus1_idxs = np.arange((Ca_idx+1)*a_per_res, (Ca_idx+1)*a_per_res+2)
            xyz[17:19] = crd_on_ref[Ca_plus1_idxs]

        # for C terminus
        else:
            Ca_minus1_idxs = np.arange((Ca_idx-1)*a_per_res+1, (Ca_idx-1)*a_per_res+4)
            xyz[:3] = crd_on_ref[Ca_minus1_idxs]

        # All Ca values
        all_Ca_idxs = np.arange(len(seq))*a_per_res + 1
        all_Ca_coords = crd_on_ref[all_Ca_idxs]

        # append xyz with whatever xyz 
        if 'Closest-Cas' in cond_type:

            # idxs sorted and add to xyz
            Ca_sorted_idxs, all_Ca_sorted = sort_Ca(all_Ca_coords, Ca_idx, 
                                    n_keep=n_cond_res, exclude_neighbors=exclude_neighbors)
            xyz = np.concatenate([xyz, all_Ca_sorted])

            # 20-dim ohe
            ohe = encode_seq(seq[Ca_idx])[0]

            # if adding a sequence position encoding
            if 'tanh' in cond_type:
                a_pos = 20 
                relative_pos = Ca_sorted_idxs - Ca_idx
                tanh_pos = np.tanh(relative_pos / a_pos)

                # add tanh_positions after between flattened Ca and ohe
                xyz = np.concatenate([xyz.flatten(), tanh_pos, ohe])
            else:
                xyz = np.concatenate([xyz.flatten(), ohe])

        elif 'Closest-scs-N2C' in cond_type:

            # idxs sorted either by Ca distance or dist to nearest sidechain
            if 'resdist' in cond_type:

                Ca_sorted_idxs, all_Ca_sorted = sort_Ca_all_sc(crd_on_ref, all_Ca_coords, 
                                                           Ca_idx, n_keep=n_cond_res, com_shift=com_shift, 
                                                               round2=True, ignore_ter=ignore_ter)

            else:
                Ca_sorted_idxs, all_Ca_sorted = sort_Ca(all_Ca_coords, Ca_idx, 
                                            n_keep=n_cond_res, exclude_neighbors=exclude_neighbors)

            block_size = a_per_res - 3 # for 10 sidechain + 1 Ca 
            xyz_cond = np.zeros((n_cond_res*block_size, 3))

            # try populating only one Ca line vs. all non-zero = Ca
            for i, Ca_s_idx in enumerate(Ca_sorted_idxs):

                # update Ca positions for all closest Cas
                Ca_crd_idx = Ca_s_idx*a_per_res + 1
                xyz_cond[block_size*i] = crd_on_ref[Ca_crd_idx]

                # collect all SC info if already decoded
                # or if round2, always collect all info
                if Ca_s_idx < Ca_idx or round2: # or Ca_s_idx in ter:
                    sc_idx_first, sc_idx_last = Ca_s_idx*a_per_res + 4, (Ca_s_idx+1)*a_per_res
                    xyz_cond[block_size*i+1: block_size*(i+1)] = crd_on_ref[sc_idx_first: sc_idx_last]

                # fill in blank sidechain blocks with Ca values, otherwise leave them as zeros
                elif 'fill-Cas' in cond_type:

                    for sc_pos in range(block_size-1):
                        cond_idx = block_size*i + sc_pos + 1
                        crd_idx = Ca_s_idx*a_per_res + sc_pos + 4

                        # break if hit all zeros and keep remainin entries at zero
                        if np.sum(crd[crd_idx]) != 0:
                            xyz_cond[cond_idx] = crd_on_ref[Ca_crd_idx]
                        else:  break

            # if protein is shorter than just repeat last values n_cond_res - N
            last_i = len(Ca_sorted_idxs) - 1
            if last_i+1 < n_cond_res:

                if short=='all':
                    last_cond = xyz_cond[block_size*last_i: block_size*(last_i+1)] 
                    for i in range(last_i+1, n_cond_res):
                        xyz_cond[block_size*i: block_size*(i+1)] = last_cond

                elif short=='Ca':
                    last_Ca = xyz_cond[block_size*last_i] 
                    for i in range(last_i+1, n_cond_res):
                        xyz_cond[block_size*i] = last_Ca

                else:
                    pass

            # append to residue input
            xyz = np.concatenate([xyz, xyz_cond])

            # 20-dim ohe
            ohe = encode_seq(seq[Ca_idx])[0]
            xyz = np.concatenate([xyz.flatten(), ohe])

        xyz_list.append(xyz)

    return xyz_list


def save_as_test_terminal(all_seqs, all_crds, all_ters, save_dir, prefix, save_type='test',
                 cond_type='Closest-Cas', n_cond_res=10, exclude_neighbors=False, single_resid=None):
    '''For saving and decoding terminal residues only -- dont perform bond or res checks'''
    
    print('Starting alignment')
    
    test_xyzs = []
    bond_rejects = []
    res_rejects = []
              
    for i, (seq, crd, ter) in tqdm(enumerate(zip(all_seqs, all_crds, all_ters)), total=len(all_seqs)):
        
        # align and add to test list -- should pass in arbitrary condition funciton here
        xyz_list = gen_aligned_xyzs_terminal(seq, crd, ter, cond_type=cond_type, n_cond_res=n_cond_res, 
                                    exclude_neighbors=exclude_neighbors, single_resid=single_resid)
        
        test_xyzs += xyz_list
    
    # this leaves off terminal residues
    test_xyzs = np.array(test_xyzs)
    
    # this should be generally applicable to any conditioning on a 19-met input
    mask_idxs = list(np.concatenate([np.arange(i*3,i*3+3) for i in [0,4,18]])) + list(np.arange(19*3, len(xyz_list[0])))

    np.save(f'{save_dir}/{save_type}_{prefix}_traj.npy', test_xyzs)
    np.save(f'{save_dir}/test_{prefix}_shared_idxs.npy', mask_idxs)
    
    if save_type=='test':
        np.save(f'{save_dir}/{save_type}_{prefix}_cond_traj.npy', test_xyzs[:, mask_idxs])
    
    return test_xyzs


def align_to_backbone_terminal(seq, crd, ter, xyz_list, single_resid):
    '''Return crd format for an input of generated xyz coordinates by superposing on
      neiboring C or N side of backbone'''

    n_res = len(seq)
    a_per_res = 14 # this is always 14
    pred_size = 19 # asuming Ca triplet with overlaps
    
    # make a copy of crd
    crd_gen = copy.deepcopy(crd)

    # find the termini of interest
    Ca_idx = ter[single_resid]
    
    # get xyz
    xyz_gen = xyz_list[0][:pred_size*3].reshape(pred_size, 3)
        
    # for N-terminus
    if single_resid%2==0:
    
        # Ca, C, Ca+1
        backbone_idxs = np.array([1,2,a_per_res+1])
        triplet_xyz_idxs = np.array([4,5,18])

    # for C-terminus
    else:
        
        # Ca-1, C-1, Ca
        backbone_idxs = np.array([-a_per_res+1,-a_per_res+2,1])
        triplet_xyz_idxs = np.array([0,1,4])
        
    # Ca-Ca pair only
    backbone_pair_idxs = np.array([backbone_idxs[0], backbone_idxs[2]])
    pair_gen_idxs = np.array([triplet_xyz_idxs[0], triplet_xyz_idxs[2]])
        
    # get exact backbone idxs
    triplet_bb_idxs = Ca_idx*a_per_res + backbone_idxs
    triplet_bb = crd[triplet_bb_idxs]
    
    pair_bb_idxs = Ca_idx*a_per_res + backbone_pair_idxs
    pair_bb = crd[pair_bb_idxs]

    # get correpsonding gen xyz (idx starting from terminal)
    triplet_gen = xyz_gen[triplet_xyz_idxs]

    # superpose on shared backboend
    sup = SVDSuperimposer()
    sup.set(triplet_bb, triplet_gen)
    sup.run()
    rms = sup.get_rms()
    rot, tran = sup.get_rotran()

    # superpose on shared backbones
    gen_on_ref = dot(xyz_gen, rot) + tran
    
    # apply additional transformation to ensure Ca positon are same as refenrence
    gen_xyz_pair = xyz_gen[pair_gen_idxs]
    gen_on_ref_pair = gen_on_ref[pair_gen_idxs]

    Ca_pair_shift = np.mean(gen_on_ref_pair, axis=0) - np.mean(pair_bb, axis=0)
    gen_on_ref -= Ca_pair_shift
    
    # for termini, replace the residue block only
    
    sc_idxs = np.arange((Ca_idx)*a_per_res, (Ca_idx+1)*a_per_res)
    if single_resid%2==0:
        bb_idxs = [(Ca_idx+1)*a_per_res]
        gen_idxs = np.arange(3, 18)
        replace_idxs = np.concatenate([sc_idxs, bb_idxs])
    else:
        bb_idxs = np.arange((Ca_idx-1)*a_per_res+2, (Ca_idx-1)*a_per_res+4)
        gen_idxs = np.arange(1, 17)
        replace_idxs = np.concatenate([bb_idxs, sc_idxs])
    
    # only overwrite indices that will be present in each termini
    crd_gen[replace_idxs] = gen_on_ref[gen_idxs]

    # make sure to keep zero values after superpose -- but don't include zeros within residue
    zero_idxs = np.where(np.all(crd==0, axis=1))[0]

    # unmask residue values for cg trajs (overides zeros)
    n_unmskd = n_masked_dict[seq[Ca_idx]]
    unmskd_list = [n for n in range(Ca_idx*a_per_res, Ca_idx*a_per_res + n_unmskd)]
    
    # also unmask any backbone predictions for adjacent residues -- don't do this for sidechians
    unmskd_list += [(Ca_idx-1)*a_per_res+2, (Ca_idx-1)*a_per_res+3, (Ca_idx+1)*a_per_res]
    
    # remove all unmasked from zero lsit
    zero_idxs = list( set(zero_idxs).difference(set(unmskd_list)) )

    # reset to zero
    crd_gen[zero_idxs] = crd[zero_idxs]
        
    return crd_gen

### Scoring metrics ###
    
def clash_score(trj, min_cut=0.12, max_cut=0.5):

    pairs = []
    pair_names = []
    a_list = [a.name for a in trj.top.atoms]

    for i, i_name in enumerate(a_list):
        for j, j_name in enumerate(a_list[:i]):
            pairs.append([i,j])
            pair_names.append([i_name, j_name])
        
    dists = md.compute_distances(trj, pairs)
    ratio = 100*np.sum(dists < min_cut) / np.sum(dists < max_cut)
    return ratio 

def rmsd_score(trj_ref, trj_gen):
    
    rmsd_list = []
    for i, (trj_r, trj_g) in enumerate(zip(trj_ref, trj_gen)):
        rmsd = md.rmsd(trj_r, trj_g)*10
        rmsd_list.append(rmsd)
    
    return np.mean(rmsd_list)
    
def ged_score_atoms(trj_ref, trj_gen, scale=1.3):
    
    pairs = []
    pair_names = []
    a_list = [a.name for a in trj_ref.top.atoms]
    n_atoms = len(a_list)
    
    # use the same cutoff for N-C-O
    cut_arr = np.zeros((len(a_list), len(a_list))) + 0.68*2
    
    for i, i_name in enumerate(a_list):
        for j, j_name in enumerate(a_list):
            
            pairs.append([i,j])
            pair_names.append([i_name, j_name])
            
            if 'S' in i_name and 'S' in j_name:
                cut_arr[i, j] = 1.02*2
                
            elif 'S' in i_name or 'S' in j_name:
                cut_arr[i, j] = 0.68+1.02

    ref_dist = md.compute_distances(trj_ref, pairs).reshape(-1, n_atoms, n_atoms)
    gen_dist = md.compute_distances(trj_gen, pairs).reshape(-1, n_atoms, n_atoms)
    
    ref_bonds = ref_dist < cut_arr*scale
    gen_bonds = gen_dist < cut_arr*scale
    
    #diff = ref_bonds.astype(np.float32) - gen_bonds.astype(np.float32) 
    #diff_mean = np.mean(abs(diff))
    
    diff_mean = abs(np.sum(ref_bonds.astype(np.float32) - gen_bonds.astype(np.float32))) / np.sum(ref_bonds)
    
    return diff_mean
    
def ged_corrected(trj_ref, trj_gen, scale=1.3):
    
    pairs = []
    pair_names = []
    a_list = [a.name for a in trj_ref.top.atoms]
    n_atoms = len(a_list)
    
    # use the same cutoff for N-C-O
    cut_arr = np.zeros((len(a_list), len(a_list))) + 0.68*2
    
    for i, i_name in enumerate(a_list):
        for j, j_name in enumerate(a_list):
            
            pairs.append([i,j])
            pair_names.append([i_name, j_name])
            
            if 'S' in i_name and 'S' in j_name:
                cut_arr[i, j] = 1.02*2
                
            elif 'S' in i_name or 'S' in j_name:
                cut_arr[i, j] = 0.68+1.02

    ref_dist = md.compute_distances(trj_ref, pairs).reshape(-1, n_atoms, n_atoms)
    gen_dist = md.compute_distances(trj_gen, pairs).reshape(-1, n_atoms, n_atoms)
    
    ref_bonds = ref_dist < cut_arr*scale
    gen_bonds = gen_dist < cut_arr*scale
    
    diff_mean = np.sum(np.abs(ref_bonds.astype(np.float32) - gen_bonds.astype(np.float32))) / np.sum(ref_bonds)
    
    return diff_mean



def bond_rmsd(trj_ref, trj_gen):
    
    bond_pairs = [[b[0].index, b[1].index] for b in trj_ref.top.bonds]
    
    ref_dist = md.compute_distances(trj_ref, bond_pairs)
    gen_dist = md.compute_distances(trj_gen, bond_pairs)
    
    rmsd_dist = np.sqrt(np.mean((ref_dist-gen_dist)**2))*10 # convert to Angstroms
                  
    return rmsd_dist


def ref_rmsd(trj_ref, trj_sample_list):
    
    rmsd_list = []
    for i, trj_i in enumerate(trj_sample_list):
        
        print(trj_i.xyz.shape, trj_ref.xyz.shape)

        for k, (trj_if, trj_rf) in enumerate(zip(trj_i, trj_ref)):
            rmsd = md.rmsd(trj_if, trj_rf)*10
            rmsd_list.append(rmsd)
            #frame_rmsds.append(rmsd)
        #rmsd_list.append(np.mean(frame_rmsds))

    return np.mean(rmsd_list), np.std(rmsd_list)

def sample_rmsd(trj_sample_list):
    
    rmsd_list = []
    for i, trj_i in enumerate(trj_sample_list):
        for j, trj_j in enumerate(trj_sample_list[:i]):
            
            # need to compare per frames rmsds or else will be relative to first frame
            #frame_rmsds = []
            for k, (trj_if, trj_jf) in enumerate(zip(trj_i, trj_j)):
                rmsd = md.rmsd(trj_if, trj_jf)*10
                rmsd_list.append(rmsd)
                #frame_rmsds.append(rmsd)
            #rmsd_list.append(np.mean(frame_rmsds))

    return np.mean(rmsd_list), np.std(rmsd_list)

def sample_rmsd_percent(trj_ref, trj_sample_list):
    
    R_ref, S_ref = ref_rmsd(trj_ref, trj_sample_list)
    R_sam, S_sam = sample_rmsd(trj_sample_list)
    
    R_per = (R_ref-R_sam) / R_ref
    S_per = np.sqrt( (S_sam/R_ref)**2 + ((R_sam*S_ref)/(R_ref)**2)**2 )
    
    return R_per, S_per


def bond_fraction(trj_ref, trj_gen, fraction=0.1):
    '''Fraction of bonds within X percent of the reference'''

    bond_pairs = [[b[0].index, b[1].index] for b in trj_ref.top.bonds]
    ref_dist = md.compute_distances(trj_ref, bond_pairs)
    gen_dist = md.compute_distances(trj_gen, bond_pairs)

    bond_frac = np.sum((gen_dist < (1+fraction)*ref_dist) & 
                       (gen_dist > (1-fraction)*ref_dist))

    bond_frac = bond_frac / np.size(ref_dist)
    
    return bond_frac

def bond_fraction_auc(trj_ref, trj_gen, frac_list=None):
    '''bond_fraction integrated across a range of fraction cutoffs'''
    
    # use 0-20% by default
    if frac_list == None:
        frac_list = np.arange(0,21)/100
    
    bond_frac_list = []
    for fraction in frac_list:
        bond_frac = bond_fraction(trj_ref, trj_gen, fraction=fraction)
        bond_frac_list.append(bond_frac)
        
    return np.sum(bond_frac_list) / (np.max(frac_list) - np.min(frac_list))