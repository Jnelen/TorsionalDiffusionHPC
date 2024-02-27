from argparse import ArgumentParser
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import pickle
import pandas as pd
from tqdm import tqdm
import yaml
import os.path as osp
import time
import warnings
import torch
import numpy as np
import gzip
import math

from utils.utils import get_model
from diffusion.sampling import *

## Parse arguments
parser = ArgumentParser()
parser.add_argument('--model_dir', type=str, required=True, help='Path to folder with trained model and hyperparameters')
parser.add_argument('--ckpt', type=str, default='best_model.pt', help='Checkpoint to use inside the folder')
parser.add_argument('--out', type=str, help='Path to the output pickle file')
parser.add_argument('--test_csv', type=str, default='./data/DRUGS/test_smiles.csv', help='Path to csv file with list of smiles and number conformers')
parser.add_argument('--pre_mmff', action='store_true', default=False, help='Whether to run MMFF on the local structure conformer')
parser.add_argument('--post_mmff', action='store_true', default=False, help='Whether to run MMFF on the final generated structures')
parser.add_argument('--no_random', action='store_true', default=False, help='Whether avoid randomising the torsions of the seed conformer')
parser.add_argument('--no_model', action='store_true', default=False, help='Whether to return seed conformer without running model')
parser.add_argument('--seed_confs', default=None, help='Path to directly specify the seed conformers')
parser.add_argument('--seed_mols', default=None, help='Path to directly specify the seed molecules (instead of from SMILE)')
parser.add_argument('--single_conf', action='store_true', default=False, help='Whether to start from a single local structure')
parser.add_argument('--inference_steps', type=int, default=20, help='Number of denoising steps')
parser.add_argument('--limit_mols', type=int, default=None, help='Limit to the number of molecules')
parser.add_argument('--confs_per_mol', type=int, default=None, help='If set for every molecule this number of conformers is generated, otherwise 2x the number in the csv file')
parser.add_argument('--ode', action='store_true', default=False, help='Whether to run the probability flow ODE instead of the SDE')
parser.add_argument('--likelihood', choices=['full', 'hutch'], default=None, help='Technique to compute likelihood')
parser.add_argument('--dump_pymol', type=str, default=None, help='Whether to save .pdb file with denoising dynamics')
parser.add_argument('--tqdm', action='store_true', default=False, help='Whether to show progress bar')
parser.add_argument('--water', action='store_true', default=False, help='Whether to compute xTB energy in water')
parser.add_argument('--batch_size', type=int, default=32, help='Number of conformers generated in parallel')
parser.add_argument('--xtb', type=str, default=None, help='If set, it indicates path to local xtb main directory')
parser.add_argument('--no_energy', action='store_true', default=False, help='If set skips computation of likelihood, energy etc')

parser.add_argument('--pg_weight_log_0', type=float, default=None)
parser.add_argument('--pg_weight_log_1', type=float, default=None)
parser.add_argument('--pg_repulsive_weight_log_0', type=float, default=None)
parser.add_argument('--pg_repulsive_weight_log_1', type=float, default=None)
parser.add_argument('--pg_langevin_weight_log_0', type=float, default=None)
parser.add_argument('--pg_langevin_weight_log_1', type=float, default=None)
parser.add_argument('--pg_kernel_size_log_0', type=float, default=None)
parser.add_argument('--pg_kernel_size_log_1', type=float, default=None)
parser.add_argument('--pg_invariant', type=bool, default=False)

parser.add_argument('--random_seed', type=int, default=None, help='Random seed to produce (approximate) deterministic results')
parser.add_argument('--rdk_random_coords', default=False, action='store_true', help='Use the randomCoords option when generating the initial RDKit conformers (more robust, but slightly slower)')
parser.add_argument('--smiles_as_id', '-si', action='store_true', default=False, help='Use a molecule\'s smile as the ID, even if the molecule has a name')
parser.add_argument('--compress_output', action='store_true', default=False, help='Compress the output pkl files using gzip')

args = parser.parse_args()

"""
    Generates conformers for a list of molecules' SMILE given a trained model
    Saves a pickle with dictionary with the SMILE as key and the RDKit molecules with generated conformers as value 
"""

failList = []

## Ensure PyTorch uses the correct amount of cores
try:
	cpuCores = int(os.environ["SLURM_CPUS_ON_NODE"])
	torch.set_num_threads(cpuCores)
except:
	cpuCores = 0

## Initiate random seed if set
rdkSeed = -1
if args.random_seed:
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    rdkSeed = args.random_seed
    
## To suppress warnings from torch's own torch/overrides.py: 
warnings.filterwarnings("ignore", category=UserWarning)

startTime = time.time()

if args.likelihood:
    assert args.ode or args.no_model


def embed_func(mol, numConfs):
    AllChem.EmbedMultipleConfs(mol, numConfs=numConfs, numThreads=cpuCores, useRandomCoords=args.rdk_random_coords, randomSeed=rdkSeed)
    return mol


still_frames = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = args.batch_size

if args.seed_confs:
    print("Using local structures from", args.seed_confs)
    with open(args.seed_confs, 'rb') as f:
        seed_confs = pickle.load(f)
elif args.seed_mols:
    print("Using molecules from", args.seed_mols)
    with open(args.seed_mols, 'rb') as f:
        seed_confs = pickle.load(f)

with open(f'{args.model_dir}/model_parameters.yml') as f:
    args.__dict__.update(yaml.full_load(f))
args.batch_size = batch_size  # override the training one
if not args.no_model:
    model = get_model(args)
    state_dict = torch.load(f'{args.model_dir}/{args.ckpt}', map_location=torch.device('cpu'))
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()

test_data = pd.read_csv(args.test_csv, sep=";").values
if args.limit_mols:
    test_data = test_data[:args.limit_mols]

conformer_dict = {}

test_data = tqdm(enumerate(test_data), total=len(test_data), ascii=True, disable=not args.tqdm)

def sample_confs(molName, n_confs, smi):
      
    if args.seed_confs:
        mol, data = get_seed(molName, seed_confs=seed_confs, dataset=args.dataset)
    elif args.seed_mols:
        mol, data = get_seed(smi, seed_confs=seed_confs, dataset=args.dataset)
        mol.RemoveAllConformers()
    else:
        mol, data = get_seed(smi, dataset=args.dataset)
    if not mol:
        print(f'Failed to get seed for {molName} with smiles: {smi}')
        return None

    n_rotable_bonds = int(data.edge_mask.sum())
    
    if args.seed_confs:
        conformers, pdb = embed_seeds(mol, data, n_confs, single_conf=args.single_conf, smi=molName,
                                      pdb=args.dump_pymol, seed_confs=seed_confs, molName=molName)
    else:
        conformers, pdb = embed_seeds(mol, data, n_confs, single_conf=args.single_conf,
                                      pdb=args.dump_pymol, embed_func=embed_func, mmff=args.pre_mmff, molName=molName)
    if not conformers:
        print(f'Failed to embed {molName} with smiles: {smi}')
        return None

    if not args.no_random and n_rotable_bonds > 0.5:
        conformers = perturb_seeds(conformers, pdb)

    if not args.no_model and n_rotable_bonds > 0.5:
        conformers = sample(conformers, model, args.sigma_max, args.sigma_min, args.inference_steps,
                            args.batch_size, args.ode, args.likelihood, pdb,
                            pg_weight_log_0=args.pg_weight_log_0, pg_weight_log_1=args.pg_weight_log_1,
                            pg_repulsive_weight_log_0=args.pg_repulsive_weight_log_0,
                            pg_repulsive_weight_log_1=args.pg_repulsive_weight_log_1,
                            pg_kernel_size_log_0=args.pg_kernel_size_log_0,
                            pg_kernel_size_log_1=args.pg_kernel_size_log_1,
                            pg_langevin_weight_log_0=args.pg_langevin_weight_log_0,
                            pg_langevin_weight_log_1=args.pg_langevin_weight_log_1,
                            pg_invariant=args.pg_invariant, mol=mol)

    if args.dump_pymol:
        if not osp.isdir(args.dump_pymol):
            os.mkdir(args.dump_pymol)
        pdb.write(f'{args.dump_pymol}/{molName}.pdb', limit_parts=5)
    
    mols = [pyg_to_mol(mol, conf, args.post_mmff, rmsd=not args.no_energy) for conf in conformers]
    
    if args.likelihood:
        if n_rotable_bonds < 0.5:
            print(f"Skipping {molName} with {smi} because it has 0 rotable bonds")
            return None
    
    for mol, data in zip(mols, conformers):
        populate_likelihood(mol, data, water=args.water, xtb=args.xtb)

    if args.xtb:
        mols = [mol for mol in mols if mol.xtb_energy]
    return mols
    
if not args.no_energy:
    csvIdx = args.out.split('.pkl')[0].split('_')[-1]
    csvFile = open(f"{os.path.dirname(args.out)}/../energies/energyData_{csvIdx}.csv",'w')
    csvFile.write("mol_name;rot_bonds;n_confs;rmsd;F;energy;dlogp;smiles\n")

intervalThreshold = math.ceil(len(test_data)/10)
print(f"Starting TorsionalDiffusion Calculations for {len(test_data)} compounds..")

for smi_idx, (molName, n_confs, smi) in test_data:

    if args.smiles_as_id:
        molName = smi
    
    if type(args.confs_per_mol) is int:
        mols = sample_confs(molName, args.confs_per_mol, smi)
    else:
        mols = sample_confs(molName, 2 * n_confs, smi)
        
    if not mols:
        failList.append([molName, smi])
        continue
    
    if not args.no_energy:
        rmsd = [mol.rmsd for mol in mols]
        dlogp = np.array([mol.euclidean_dlogp for mol in mols])
        
        if args.xtb:
            energy = np.array([mol.xtb_energy for mol in mols])
        else:
            energy = np.array([mol.mmff_energy for mol in mols])

        ## Write energy information to csvFile (in energy folder)
        if not np.isnan(energy).any():
            F, F_std = (0, 0) if args.no_energy else free_energy(dlogp, energy)  
            csvFile.write(";".join([str(molName), f"{mols[0].n_rotable_bonds}", f"{len(rmsd)}", f"{np.mean(rmsd):.2f}", f"{F:.2f}+/-{F_std:.2f}", f"{np.mean(energy):.2f}+/-{bootstrap((energy,), np.mean).standard_error:.2f}", f"{np.mean(dlogp):.2f}+/-{bootstrap((dlogp,), np.mean).standard_error:.2f}", smi]) + "\n")
        else:
            csvFile.write(";".join([str(molName), f"{mols[0].n_rotable_bonds}", f"{len(rmsd)}", f"{np.mean(rmsd):.2f}", f"{np.nan}", f"{np.nan}", f"{np.mean(dlogp):.2f}+/-{bootstrap((dlogp,), np.mean).standard_error:.2f}", smi]) + "\n")
    
    conformer_dict[molName] = mols
    
    if smi_idx % intervalThreshold == 0 and not args.tqdm:
        print(f"Processed {smi_idx+1} molecules ({round(100*(smi_idx/len(test_data)))}%), {time.time()-startTime:.2f} seconds elapsed")
    
if not args.no_energy:
    csvFile.close()

# save to file
if args.out:
    if not args.compress_output:
        with open(f'{args.out}', 'wb') as f:
            pickle.dump(conformer_dict, f)
    else:
        with gzip.open(f'{args.out}.gz', 'wb') as f:
            pickle.dump(conformer_dict, f)
                  
print(f"\nFinished after {time.time()-startTime:.2f} seconds")       
print(f"Succesfully generated conformers for {len(conformer_dict)} molecules, {len(failList)} compounds failed")

if len(failList) > 0:
     print('The following compounds failed:')	
     print(failList)

