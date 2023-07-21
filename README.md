# DiAMoNDBack: Diffusion-denoising Autoregressive Model for Non-Deterministic Backmapping of Cα Protein Traces


A transferable approach for backmapping generic Cα protein traces using Denoising Diffusion Probabilistic Models (DDPMs).

The DDPM models contained herein are adapted from the implementation by https://github.com/lucidrains/denoising-diffusion-pytorch, borrowing also from some changes to this implementation by https://github.com/tiwarylab/DDPM_REMD. 


## Environmnet

The environment file provided at `env.yml` can be used to create the `DiffBack` environment with:

```bash
$ conda env create -f env.yml
```

The environment can then be activated with:

```bash
$ conda activate diffback
```

## Performing backmapping with pre-trained models

Inference can be performed using a pre-trained or user-trained model via the `run_eval.py` script. To use one of our pre-trained model, specify PDB or PDB_DES-FT in the `--training_set` argument. Users can reference custom models based on the name of the training set and hyperparameter values. The trajectory stride as well as the number of distinct generated replicas can also be specified. Several examples are shown below referencing models and data sets used in the paper:

To backmap the PDB test data using the PDB-trained model:
```bash
$ python run_eval.py --training_set PDB --data_type aa --pdb_dir ../data/all_train_test_pdbs/PDB_test
```

To backmap the DES test set using the DES-finetuned model:
```bash
python run_eval.py --training_set PDB_DES-FT --data_type aa --pdb_dir ../data/all_train_test_pdbs/DES_test
```

Same as above, but use a stride of 100 and generate 3 distinct samples per frame:
```bash
python run_eval.py --training_set PDB_DES-FT --data_type aa --pdb_dir ../data/all_train_test_pdbs/DES_test --n_samples 3 --stride 100
```

## Training

We provide the functionality to train DDPM backmapping models both from scratch and fine-tune pre-trained models. We recommend fine-tuning models starting from the pre-trained model provided in `pre_trained_models/PDB_trained`. This pre-trained model was trained on 65k+ PDB structures and serves as a good starting point for building bespoke backmapping models on possibly small amounts of available atomistic data. 


### Fine-tuning

To fine-tune begin with compiling a directory containing atomistic PDB structures that will be used for fine-tuning, for example in `data/train_pdbs`. Then, with the environment activated, navigate to the scripts directory and execute the training script specifying the pre-trained model path for fine-tuning. Specifying PDB or PDB_DES-FT will begin fine-tuning respectively from the PDB-trained or DES-finetuned models discussed in the paper. To fine-tune on a new model, specify the full model path in the finetune argument.

```bash
$ cd scripts
$ python train.py --pdb_dir ../data/train_pdbs/ --save_name test_run --finetune PDB
```

The model will begin training and periodically output checkpoints to a `./trained_models/train_{save_name}` directory. The argument provided to `--pdb_dir` should be a directory containing PDB files that will by default be processed for featurization and saved to `data/processed_features`. As featurization can become costly for large quantities of structures, this preprocessing step can be skipped by providing the `--run_preprocess 0` flag in which case the existing feature files with the associated `save_name` contained in `data/processed_features` will be used. 

### Training from scratch 

With a directory containing PDB structures that will be used for training, for example in `data/train_pdbs`, the same training script as for fine-tuning can be run simply without the `--finetune` argument:

```bash
$ cd scripts
$ python train.py --pdb_dir ../data/train_pdbs/ --save_name test_run 
```

Model checkpoints will start saving to a `./trained_models/train_{save_name}` directory, with the same options for handling feature preprocessing available for training from scratch as discussed in the fine-tuning routines. 

## Data splits

Training and evaluation splits in pdb format for all data used to train and test the models presented in the paper are available for download via [Zenodo](https://zenodo.org/record/8169239).
