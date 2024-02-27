#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on Fri Feb 2 20:46:07 2024

@author: Jochem Nelen (jnelen@ucam.edu)
"""

import datetime
import glob
import itertools
import os
import shutil
import subprocess
import sys
import time

import argparse
from argparse import ArgumentParser, RawTextHelpFormatter

## Define the arguments
parser = ArgumentParser(description="Slurm based job launcher for Torsional Diffusion", formatter_class=RawTextHelpFormatter)

torsionalDiffusion_group = parser.add_argument_group('TorsionalDiffusion Options')
extra_group = parser.add_argument_group('Extra Options')
slurm_group = parser.add_argument_group('Slurm Options')

torsionalDiffusion_group.add_argument('--ligands', '-l', required=True, type=str, default='', help='The path to and sdf file or a directory of mol(2)/sdf ligand files')
torsionalDiffusion_group.add_argument('--out_dir', '-out', '-o', required=True,type=str, default='', help='Directory where the output structures will be saved to')
torsionalDiffusion_group.add_argument('--num_confs', '-n', type=int, default=10, help='How many conformers to output per compound. The default value is 10')
torsionalDiffusion_group.add_argument('--inference_steps', '-is', type=int, default=20, help='Number of denoising steps')
torsionalDiffusion_group.add_argument('--batch_size', '-bs', type=int, default=32, help='Number of conformers generated in parallel')
torsionalDiffusion_group.add_argument('--model_dir', '-md', type=str, default="workdir/drugs_default/", help='Path to folder with trained model and hyperparameters')

torsionalDiffusion_group.add_argument('--tqdm', '-tqdm', action='store_true', default=False, help='Use the more verbose tqdm to show progress for every compound.')
torsionalDiffusion_group.add_argument('--dump_pymol', action='store_true', default=False, help='Save .pdb file with denoising dynamics')
torsionalDiffusion_group.add_argument('--pre-mmff', action='store_true', default=False, help='Run MMFF on the local structure conformer')
torsionalDiffusion_group.add_argument('--post-mmff', action='store_true', default=False, help='Run MMFF on the final generated structures')
torsionalDiffusion_group.add_argument('--no_energy', '-ne', action='store_true', default=False, help='Skip Calculating the energies and other metrics')
torsionalDiffusion_group.add_argument('--particle_guidance', '-pg', type=int, choices=[0,1,2,3,4], default=0, help='Define which type of Particle Guidance you want to use:\n0: No particle guidance\n1: Permutation invariant minimize recall error\n2: Permutation invariant minimize precision error\n3: Non-permutation invariant minimize recall error\n4: Non-permutation invariant minimize precision error')

extra_group.add_argument('--smiles_as_id', '-si', action='store_true', default=False, help='Use a molecule\'s smile as the ID, even if the molecule has a name')
extra_group.add_argument('--compress_output', '-co', action='store_true', default=False, help='Compress the output pkl files using gzip')
extra_group.add_argument('--remove_salts', '-rs', action='store_true', default=False, help='Remove salts and fragments from the input molecules')
extra_group.add_argument('--random_coords', '-rc', action='store_true', default=False, help='Use the "useRandomCoords=True" option when generating initial RDKit conformers (more robust, but slower)')
extra_group.add_argument('--random_seed', '--seed', type=int, default=None, help='Random seed to produce (approximate) deterministic results')

slurm_group.add_argument('--jobs', '-j', required=True, type=int, default=1, help='Number of jobs to use')
slurm_group.add_argument('--time', '-t', '-tj', required=False, default="", help='Max amount of time each job can run')
slurm_group.add_argument('--queue', '-qu', type=str, default="", help='On which node to launch the jobs. The default value is the default queue for the user. Might need to be specified if there is no default queue configured')
slurm_group.add_argument('--mem', '-m', type=str, default="4G", help='How much memory to use for each job. The default value is 4GB')
slurm_group.add_argument('--gpu', '-gpu', '-GPU', '--GPU', action="store_true", default=False, help='Use GPU resources. This will accelerate docking calculations if a compatible GPU is available')
slurm_group.add_argument('--cores', '-c', type=int, default=1, help='How many cores to use for each job. The default value is 1')

args = parser.parse_args()

## Check if Singularity image is present
if not os.path.exists("singularity/TorsionalDiffusionHPC.sif"):
	print("The Singularity image doesn't seem to be present. Please follow the installation instructions and download it from there, or build it manually using the def file")
	sys.exit()
	
outputPath, outputDirName = os.path.split(args.out_dir)

currentDateNow = datetime.datetime.now()

if not outputPath == "":
	outputPath += "/"

## Make final output path
if args.particle_guidance == 0:
	pgStr = "NoPG"
else:
	pgStr = f"PG-{args.particle_guidance}"
	
outputDir = outputPath + "_".join([outputDirName, f"Confs{args.num_confs}", pgStr, str(currentDateNow.year), str(currentDateNow.month), str(currentDateNow.day)])

## Check if the output directory already exists, and asks the user what to do if it does
if os.path.isdir(outputDir):
	print(f"The directory {outputDir} already exists. To continue you must delete this directory or choose another outputname.")
	answer = input("Do you want to remove it? (y/n) ").lower()
	while answer not in ("y", "n", "yes", "no"):
		print("Invalid input. Please enter y(es) or n(o).")
		answer = input("Do you want to remove or overwrite it? (y/n) ").lower()
	if answer == "y" or answer == "yes":
		shutil.rmtree(outputDir)			
	else:
		sys.exit()
			
os.mkdir(outputDir)

os.mkdir(outputDir + "/molecules")
os.mkdir(outputDir + "/input_csvs")
os.mkdir(outputDir + "/jobs_out")
os.mkdir(outputDir + "/jobs")

print("Reading and processing input files..")
subprocess.run(f"singularity run --bind $PWD singularity/TorsionalDiffusionHPC.sif python prep_input.py {args.ligands} {outputDir + '/input_csvs'} {args.num_confs} {args.jobs} {args.remove_salts}", shell=True)

print("Launching jobs now..")	
inputFiles = sorted(glob.glob(outputDir + "/input_csvs/*.csv"), key=lambda x: int(x.split("_")[-1].split(".")[0]))

queueArgument = ""
if not args.queue == "":
	queueArgument = f" -p {args.queue}"

if args.time == "":
	timeArg = ""
else:
	timeArg = f" --time {args.time} "

if args.particle_guidance == 1:
	pgArg = "--pg_invariant=True --pg_kernel_size_log_0=1.7565691770646286 --pg_kernel_size_log_1=1.1960868735428605 --pg_langevin_weight_log_0=-2.2245183818892103 --pg_langevin_weight_log_1=-2.403905082248579 --pg_repulsive_weight_log_0=-2.158537381110402 --pg_repulsive_weight_log_1=-2.717482077162461 --pg_weight_log_0=0.8004013644746992 --pg_weight_log_1=-0.9255658381081596"
elif args.particle_guidance == 2:
	pgArg = "--pg_invariant=True --pg_kernel_size_log_0=-0.9686202580381296 --pg_kernel_size_log_1=-0.7808409291022302 --pg_langevin_weight_log_0=-2.434216242826782 --pg_langevin_weight_log_1=-0.2602238633333869 --pg_repulsive_weight_log_0=-2.0439285313973237 --pg_repulsive_weight_log_1=-1.468234554877924 --pg_weight_log_0=0.3495680598729498 --pg_weight_log_1=-0.22001939454654185"
elif args.particle_guidance == 3:
	pgArg = "--pg_kernel_size_log_0=2.35958 --pg_kernel_size_log_1=-0.78826 --pg_langevin_weight_log_0=-1.55054 --pg_langevin_weight_log_1=-2.70316 --pg_repulsive_weight_log_0=1.01317 --pg_repulsive_weight_log_1=-2.68407 --pg_weight_log_0=0.60504 --pg_weight_log_1=-1.15020"
elif args.particle_guidance == 4:
	pgArg = "--pg_kernel_size_log_0=1.29503 --pg_kernel_size_log_1=1.45944 --pg_langevin_weight_log_0=-2.88867 --pg_langevin_weight_log_1=-2.47591 --pg_repulsive_weight_log_0=-1.01222 --pg_repulsive_weight_log_1=-1.91253 --pg_weight_log_0=-0.16253 --pg_weight_log_1=0.79355"
else:
	pgArg = ""

if not args.no_energy:
	os.mkdir(outputDir + "/energies")
	energyArg = ""
else:
	energyArg = "--no_energy"
	
if args.dump_pymol:
	pymolPath = outputDir + "/pdbs/"
	os.mkdir(pymolPath)
	pymolArg = f"--dump_pymol {pymolPath}"
else:
	pymolArg = ""

tqdmArg = "--tqdm" if args.tqdm else ""
preMMFFArg = "--pre_mmff" if args.pre_mmff else ""
postMMFFArg = "--post_mmff" if args.post_mmff else ""

seedArg = f"--random_seed {args.random_seed}" if args.random_seed else ""
randomCoordsArg = "--rdk_random_coords" if args.random_coords else ""
compressOutputArg = "--compress_output" if args.compress_output else ""
smilesAsIDArg = "--smiles_as_id" if args.smiles_as_id else ""


for i, inputFile in enumerate(inputFiles):

	## Execute command using singularity and sbatch wrap giving the csv as an input, and passing the input variables as well
	if args.gpu == True:
		jobCMD = f'sbatch --wrap="singularity run --nv --bind $PWD singularity/TorsionalDiffusionHPC.sif python3 -u generate_confs.py --test_csv {inputFile} --inference_steps {args.inference_steps} --batch_size {args.batch_size} --confs_per_mol {args.num_confs} --model_dir {args.model_dir} --out {outputDir}/molecules/outputDict_{i+1}.pkl {energyArg} {pymolArg} {preMMFFArg} {postMMFFArg} {seedArg} {randomCoordsArg} {compressOutputArg} {smilesAsIDArg} {tqdmArg} {pgArg}" --mem {args.mem} --output={outputDir}/jobs_out/job_{i+1}_%j.out --gres=gpu:1 --job-name=TorsDiffHPC -c {args.cores} {timeArg} {queueArgument}'
	else:
		jobCMD = f'sbatch --wrap="singularity run --bind $PWD singularity/TorsionalDiffusionHPC.sif python3 -u generate_confs.py --test_csv {inputFile} --inference_steps {args.inference_steps} --batch_size {args.batch_size} --confs_per_mol {args.num_confs} --model_dir {args.model_dir} --out {outputDir}/molecules/outputDict_{i+1}.pkl {energyArg} {pymolArg} {preMMFFArg} {postMMFFArg} {seedArg} {randomCoordsArg} {compressOutputArg} {smilesAsIDArg} {tqdmArg} {pgArg}" --mem {args.mem} --output={outputDir}/jobs_out/job_{i+1}_%j.out --job-name=TorsDiffHPC -c {args.cores} {timeArg} {queueArgument}'
	
	with open(f"{outputDir}/jobs/job_{inputFile.split('_')[-1].split('.')[0]}.sh", "w") as jobfile:
		jobfile.write("#!/usr/bin/env bash\n")
		jobfile.write(jobCMD)
	subprocess.run(jobCMD, shell=True)
