import sys
sys.path.append('../denoising_diffusion_pytorch/')
from utils import *
import uuid

from argparse import ArgumentParser
parser = ArgumentParser()

'''
Examples

To backmap PDB test data using the PDB-trained model:
python run_eval.py --training_set PDB --data_type aa --pdb_dir ../data/all_train_test_pdbs/PDB_test

To backmap the DES test set using the DES-finetuned model:
python run_eval.py --training_set PDB_DES-FT --data_type aa --pdb_dir ../data/all_train_test_pdbs/DES_test

Same as above, but use a stride of 100 and generate distinct 3 samples per frames
python run_eval.py --training_set PDB_DES-FT --data_type aa --pdb_dir ../data/all_train_test_pdbs/DES_test --n_samples 3 --stride 100

'''

parser.add_argument("--n_samples", type=int, default=1)
parser.add_argument("--training_set", type=str, default='PDB')
parser.add_argument("--data_type", type=str, default='cg')
parser.add_argument("--pdb_dir", type=str, default='../data/all_train_test_pdbs/PDB_test')
parser.add_argument("--top", type=str, default=None)
parser.add_argument("--stride", type=int, default=1)
parser.add_argument("--decode_termini", type=str, default='True')
parser.add_argument("--gamma", type=float, default=1.0)
parser.add_argument("--train_lr", type=float, default=1.0e-05)
args = parser.parse_args()

# number of samples to generate per trajectory/frame
n_samples = args.n_samples

# name of directory where model is stored
training_set = args.training_set  # PDB_DES-FT for fine-tuned

# 'aa' or 'cg' depending on if the data are all-atom or Ca-traces only
data_type = args.data_type

# directory containing all pdbs or xtcs for testing
pdb_dir = args.pdb_dir

# path to pdb topology if loading in xtcs
top = args.top
#if top == 'None': top = None

# stride at which to generate data
stride = args.stride

# name dir
pdb_name_list = glob.glob(f'{pdb_dir}/*')

# creates a save-name list accounting for the number of gens
pdb_save_names = [s for s in [i.split('/')[-1][:-4] for i in pdb_name_list]*n_samples]

# whether or not to decode termini in addition to internal residues
decode_termini = args.decode_termini
if decode_termini=='True': decode_termini = True
else: decode_termini = False

# for pre-trained models set training params:
if training_set=='PDB': gamma, train_lr = 0.9999965988084024, 2.0e-05
elif  training_set=='PDB_DES-FT': gamma, train_lr = 1.0, 1.0e-05
else: gamma, train_lr = args.gamma, args.train_lr

# the following are hyperparameters should remain fixed in most caesse
rescale = 20.0
adamw = None
milestone = None # 3
dtime = 50
n_cond_res=14
cond_type = 'Closest-scs-N2C-resdist-acenter-rescheck-ignT-TER_N-14' 
model_name = 'Unet1D_1c_2000001s_50t' 
exclude_neighbors = False          
n_train_steps = 1_000_001


def load_trainer(system_name, model_name,  last_milestone=None, 
                 model_train_steps = 2_000_000, dtime = 50, init_dim = 32, dim_mults = (1, 2, 4, 8), train_batch_size = 128, 
                 train_lr=1e-5, gamma=None, adamw=False, rescale=None):
    
    train_name = f'train_{system_name}'
    loss = 'l1'
    beta = 'cosine'
    
    model = Unet1D(dim = init_dim, dim_mults = dim_mults, channels=1)
    
    # load shared idxs for mask -- this should be the same as results folder but that's not generated until below
    
    # mask and op_num are ./results/system/MODEL_INFO
    
    # path set by model params
    MODEL_INFO = f"{init_dim }-"
    for w in dim_mults:
        MODEL_INFO += f"{w}-"
    MODEL_INFO += f"b{train_batch_size}-"
    MODEL_INFO += f"lr{train_lr:.1e}"
    if gamma is not None:
        MODEL_INFO += f"-gamma{gamma}-"
    if adamw:
        MODEL_INFO += f"-optAdamW-"
    if rescale is not None:
        MODEL_INFO += f"-rescale{rescale}-"
       
    model_path = Path(f"./trained_models/{train_name}/{MODEL_INFO}")

    # take code from ddpm to generate this
    #model_path = '/project/andrewferguson/DiffBack/scripts/results/train_train_pdb_run/32-1-2-4-8-b128-lr3.0e-04-gamma0.9999965988084024-/'
    
    shared_idxs = np.load(f'{model_path}/shared_idxs.npy')
    op_number = np.load(f'{model_path}/op_number.npy')
    
    model = nn.DataParallel(model)
    model.to(device)
    
    diffusion = GaussianDiffusion(
        model,                          # U-net model
        timesteps = dtime,               # number of diffusion steps
        loss_type = loss,              # L1 or L2
        beta_schedule = beta,
        unmask_list = shared_idxs,
    ).to(device)
    
    #set training parameters
    trainer = Trainer(
        diffusion,                                   # diffusion model
        folder = None,                               # don't need this for inference
        system = train_name,
        train_batch_size = 128,                      # training batch size
        train_lr = train_lr,                             # learning rate
        train_num_steps = model_train_steps,         # total training steps
        gradient_accumulate_every = 1,               # gradient accumulation steps
        ema_decay = 0.995,                           # exponential moving average decay
        op_number = op_number,
        fp16 = False,
        scheduler_gamma=gamma,
        adamw=adamw,
        save_name = model_name,
        rescale=rescale# turn on mixed precision training with apex
    )
    
    # find last milestone
    trainer.RESULTS_FOLDER = Path(f'./{trainer.RESULTS_FOLDER}')
    model_list = sorted(glob.glob(f'{trainer.RESULTS_FOLDER}/model-*.pt'))
    print(trainer.RESULTS_FOLDER, len(model_list))
    
    if last_milestone is None:
        last_milestone = max([int(m.split('/')[-1].replace('.pt', '').replace('model-', '')) for m in model_list])
    print(f'Milestone = {last_milestone}')
    # start load last
    trainer.load(last_milestone)
    return trainer, last_milestone

# MJ -- use iter_gens as a place to store intermiediate configs as theyre being iteratively generated

def iterative_inference(data, system_name, cond_type, n_cond_res, model_name, pdb_save_names=[], dtime=50,
                       test_save_dir='./gen_outputs/', milestone=None, exclude_neighbors=False, 
                       n_train_steps=1_000_001,init_dim=32, dim_mults=(1,2,4,8), train_lr=1e-5,
                       gamma=None, adamw=False, rescale=None, desres=False, round2=False, short='zero', decode_termini=False):
    '''Perform interative inference
       By default uses most recent milestone'''
        
    (all_seqs, all_crds, all_ters, end_idx_list, n_sample_list, n_res_list) = data
    
    # save original copy as all_crds will be iteratively updated
    all_crds_ref = copy.deepcopy(all_crds)
    a_per_res = 14
    
    # ensures simulatenous inference files don't contaminate
    test_iter_prefix = str(uuid.uuid4())

    # load trainer -- automatically selects model with most training 
    trainer, milestone = load_trainer(system_name, model_name, dtime=dtime,
                                      model_train_steps = n_train_steps, last_milestone=milestone,
                                     init_dim=init_dim, dim_mults=dim_mults, train_lr=train_lr, gamma=gamma, adamw=adamw, rescale=rescale)

    iter_gen_xyzs = []
    for i in range(max(n_res_list)):

        # only pass in frames with n_res < i
        s_idxs = np.where(n_res_list > i)[0]

        # only include crds from sequeces longer than i -- len(rem_seqs) gradually decreases over time
        rem_seqs, rem_crds, rem_ters = [all_seqs[j] for j in s_idxs], [all_crds[j] for j in s_idxs], [all_ters[j] for j in s_idxs]

        test_xyzs, mask_idxs, _, _ = save_as_test(rem_seqs, rem_crds, rem_ters, test_save_dir, test_iter_prefix,
                            save_type='test', cond_type=cond_type, n_cond_res=n_cond_res, single_resid=i,
                            short=short, remove_zero_res=False)

        # run on ith residue for all sequences -- returns value in the aligned reference frame
        gen_xyzs = run_inference(trainer, data_name=test_iter_prefix, folder_name=test_save_dir, batch_size=5000)

        # update 14 positions in each all_crds with new in test_xzys
        for g_idx, s_idx in enumerate(s_idxs):

            # align the new residue to the previous crd backbone
            seq, crd, ter = all_seqs[s_idx], all_crds[s_idx], all_ters[s_idx]

            # get zero idxs from resid instead of all=0 
            new_crd = align_to_backbone(seq, crd, ter, [gen_xyzs[g_idx]], single_resid=i)

            # replace crd -- need to be in same ref frame
            all_crds[s_idx] = new_crd

        # collect xyzs
        iter_gen_xyzs.append(test_xyzs)
        print(f'Finished Residue: {i} / {max(n_res_list)} | Completed : {len(n_res_list)-len(s_idxs)} / {len(n_res_list)} frames\n')
        
    # MJ -- is there an issue using iter_gens here too?

    #save_dir = f'./iter_gens/{test_iter_prefix}_ckp-{last_milestone}'
    save_dir = f"./gen_outputs/{str(trainer.RESULTS_FOLDER).split('trained_models')[1]}_ckp-{milestone}"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
 
    # perform terminal inference loop
    if decode_termini:
        
        # track how many termini must be decoded by sequence
        n_ter_list = np.array([len(ters) for ters in all_ters])
        max_ters = np.max(n_ter_list)
    
        # iterate through alll remaining termini -- need to general this for all
        for i in range(max_ters):
            
            # only pass in frames with n_res < i
            s_idxs = np.where(n_ter_list > i)[0]
            
            # only include crds from sequeces longer than i -- len(rem_seqs) gradually decreases over time
            rem_seqs, rem_crds, rem_ters = [all_seqs[j] for j in s_idxs], [all_crds[j] for j in s_idxs], [all_ters[j] for j in s_idxs]
            test_xyzs = save_as_test_terminal(rem_seqs, rem_crds, rem_ters, test_save_dir, test_iter_prefix, 
                                save_type='test', cond_type=cond_type, n_cond_res=n_cond_res, single_resid=i)
            
            # run on ith residue for all sequences -- returns value in the aligned reference frame
            gen_xyzs = run_inference(trainer, data_name=test_iter_prefix, folder_name=test_save_dir, batch_size=5000)
           
            # update 14 positions in each all_crds with new info from each test_xyzs -- can we parallize this?
            for g_idx, s_idx in enumerate(s_idxs):
                # align the new residue to the previous crd backbone
                seq, crd, ter = all_seqs[s_idx], all_crds[s_idx], all_ters[s_idx]
                # get zero idxs from resid instead of all=0 -- should pass in Ca directly here? and do unmask in func?
                new_crd = align_to_backbone_terminal(seq, crd, ter, [gen_xyzs[g_idx]], single_resid=i)
                # replace crd -- need to be in same ref frame
                all_crds[s_idx] = new_crd      
    
    # convert back to mdtraj format and save
    crd_to_mdtrj(all_seqs, all_crds_ref, all_crds, end_idx_list,
                pdb_save_names, n_sample_list=n_sample_list, save_dir=save_dir)
    
    # remove intermediate files
    for f in glob.glob(f'./gen_outputs/*{test_iter_prefix}*'):
        os.remove(f)

# load pdbs in scnet format
data = prepare_pdbs(pdb_name_list, n_samples=n_samples, data_type=data_type, top=top, stride=stride)

#data = prepare_pdbs(pdb_name_list, n_samples)
iterative_inference(data, training_set, cond_type, n_cond_res, model_name, pdb_save_names=pdb_save_names, dtime=dtime,
                          milestone=milestone, exclude_neighbors=exclude_neighbors, n_train_steps=n_train_steps,
                            gamma=gamma, train_lr=train_lr, adamw=adamw, rescale=rescale, round2=False, decode_termini=True)

