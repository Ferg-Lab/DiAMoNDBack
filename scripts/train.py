from utils import *
import sys, os
from argparse import ArgumentParser

sys.path.append('../denoising_diffusion_pytorch')

from denoising_diffusion_pytorch_backmap_combined import Unet1D, GaussianDiffusion, Trainer, Dataset_traj, cycle, num_to_groups

import torch
from torch import nn 

'''
Pre-process and train on pdbs from a directory pdb_dir
If pre-processing has already been completed, set run_preprocess=False
Features will be loaded from {feat_dir}/{save_name}*

Sample input:
python train_new.py -d ../data/test_pdbs -s train_pdb_run --run_preprocess true    # to generate features
python train_new.py -d ../data/test_pdbs -s train_pdb_run --run_preprocess false   # if features are already generated

'''

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = ArgumentParser()
parser.add_argument("-d", "--pdb_dir", type=str)
parser.add_argument("-s", "--save_name", type=str)
parser.add_argument("-f", "--feat_dir", type=str, default='../data/processed_features')
parser.add_argument("-c", "--finetune", type=str, default=None)
parser.add_argument("-a", "--aug", type=int, default=None)

#parser.add_argument("-p", "--run_preprocess", type=str, default=None)    # do we want this to be a bool input?
parser.add_argument("--run_preprocess", type=str2bool, nargs='?',
                        const=True, default=True)


args = parser.parse_args()
pdb_dir = args.pdb_dir
feat_dir = args.feat_dir
save_name = args.save_name
run_preprocess = args.run_preprocess
finetune = args.finetune
aug = args.aug

# local environment size
n_cond_res = 14

if finetune is not None:
    if aug is None:
        print("Setting aug=5 as the default for fine-tuning")
        aug = 5

# provide option to retrain from pre-saved dataset (check if save already present)
if run_preprocess:
    
    try:
        os.mkdir(feat_dir)
    except:
        pass

    all_seqs, all_crds, all_ters = [], [], []

    # aggregate cross all pdbs in list
    pdb_list = glob.glob(f'{pdb_dir}/*pdb')
    for pdb in pdb_list:

        trj = md.load(pdb)
        print(pdb, trj.xyz.shape)

        seq_train, crd_train, ter_train = mdtrj_to_scnet(trj)
        print(len(seq_train), len(crd_train), len(ter_train))

        all_seqs += seq_train
        all_crds += crd_train
        all_ters += ter_train

    # don't exclude any training data based on sidechains or bonds
    xyzs, _, _, _ = save_as_test(all_seqs, all_crds, all_ters, 
                 feat_dir, save_name, save_type='train', 
                 cond_type='Closest-scs-N2C-resdist-acenter-ignT-TER',
                 n_cond_res=n_cond_res, augment_TER=aug)
    print(xyzs.shape)

# start training 

train_name = f'train_{save_name}' 
shared_name = f'test_{save_name}_shared_idxs'


if finetune is not None:
    model_train_steps = 1_500_000
    train_lr = 1e-5
    scheduler_gamma = 1.0
else:
    model_train_steps = 1_000_000
    train_lr = 3e-4
    scheduler_gamma = 0.9999965988084024031015529430944845596851594073932809450553056651


model = Unet1D(dim = 32,
               dim_mults=(1, 2, 4, 8),
               channels=1,     
               self_condition=False,
               random_fourier_features=False, 
               learned_sinusoidal_cond=False,
               learned_sinusoidal_dim=16) 

model = nn.DataParallel(model)
model.to(device)

# laod shared idxs for mask
shared_idxs = np.load(f'./{feat_dir}/{shared_name}.npy')
op_number = np.load(f'./{feat_dir}/{train_name}_traj.npy').shape[-1]

diffusion = GaussianDiffusion(
    model,                          # U-net model
    timesteps = 50,               # number of diffusion steps
    loss_type = 'l1',
    beta_schedule = "cosine",
    unmask_list = shared_idxs,
).to(device) 

#set training parameters
trainer = Trainer(
    diffusion,                                   # diffusion model
    folder = feat_dir,                        # folder of features
    system = train_name,         
    train_batch_size = 128,         # training batch size
    train_lr = train_lr,                             # learning rate
    train_num_steps = model_train_steps,       # total training steps
    gradient_accumulate_every = 1,               # gradient accumulation steps
    ema_decay = 0.995,                           # exponential moving average decay
    op_number = op_number,
    fp16 = False,
    scheduler_gamma=scheduler_gamma,
    adamw=False,
    rescale=20.0,
    save_name = save_name, 
)
    
if finetune is not None:
    print(f'Loading model weights from: {finetune}')
    trainer.load_restart(finetune)

# start training
trainer.train()