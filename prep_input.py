#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on Fri Feb 2 20:57:18 2024

@author: Jochem Nelen (jnelen@ucam.edu)
"""

import glob
import gzip
import os
import pickle
import sys

from rdkit import Chem
from tqdm import tqdm

args = sys.argv

ligandsPath = args[1]
outputDir = args[2]
nConfs = args[3]
nJobs = int(args[4])
removeSalts = True if args[5].lower() == "true" else False

def removeSaltsFromSmiles(inputSmiles):
	return max(inputSmiles.split("."), key=lambda x: len(x))
	
def processInputFile(inputPath, disableTqdm=True, removeSalts=True):
	output = []
	
	## Check file extension: .gz, .mol2, .sdf, mol, .pkl
	fileExt = inputPath.split(".")[-1].lower()

	if fileExt == "csv":
		with open(inputPath) as inputFile:
			return(inputFile.readlines()[1:])
	
	if fileExt == "gz":
		try:			
			fileExt = inputPath.split(".")[-2].lower()
			inputPath = gzip.open(inputPath)
		except:
			print(f"Error: Couldn't process {inputPath}")
			return output

	## Process pickled dicts, lists or molecules
	if fileExt == "pkl":
		try:
			inputPickle = pickle.load(inputPath)
	
			if type(inputPickle) is dict:
			
				## Check if the values are molecules or smiles			
				if type(inputPickle.values()[0]) is str:
					## Process as smiles
					for key in tqdm(inputPickle.keys(), disable=disableTqdm):
						molSmiles = inputPickle[key]
						if removeSalts:
							molSmiles = removeSaltsFromSmiles(molSmiles)
						output.append(f"{key};{nConfs};{molSmiles}\n")
				else:
					## Process as molecules
					for key in tqdm(inputPickle.keys(), disable=disableTqdm):
						try:
							molSmiles = Chem.MolToSmiles(inputPickle[key])
							if removeSalts:
								molSmiles = removeSaltsFromSmiles(molSmiles)
							output.append(f"{key};{nConfs};{molSmiles}\n")
						except:
							print(f"Error: Couldn't process {key} from {inputPath}")
							continue					
	
			elif type(inputPickle) is list:
				if type(inputPickle[0]) is str:
				## Process as smiles
					for smiles in tqdm(inputPickle, disable=disableTqdm):
						output.append(f"{smiles};{nConfs};{smiles}\n")
				else:
					## Process as molecules
					for i, mol in enumerate(tqdm(inputPickle, disable=disableTqdm)):
						try:
							molSmiles = Chem.MolToSmiles(mol)

							if removeSalts:
								molSmiles = removeSaltsFromSmiles(molSmiles)
								
							if not mol.GetProp("_Name") == "":
								molName = mol.GetProp("_Name")
							else:
								molName = molSmiles							
							output.append(f"{molName};{nConfs};{molSmiles}\n")
						except:
							print(f"Error: Couldn't process molecule {i} ({mol}) from {inputPath}")
							continue					

			## Assume it's an RDKit molecule
			else:
				try:
					molSmiles = Chem.MolToSmiles(mol)
					if removeSalts:
						molSmiles = removeSaltsFromSmiles(molSmiles)
					if not mol.GetProp("_Name") == "":
						molName = mol.GetProp("_Name")
					else:
						molName = molSmiles							
					output.append(f"{molName};{nConfs};{molSmiles}\n")
				except:
					print(f"Error: Couldn't process {inputPath}")
					return output
				
		except:
			print(f"Error: Couldn't process {inputPath}")
			return output
			
	elif fileExt == "mol2":
		mol = Chem.MolFromMol2File(inputPath)
		if not mol is None:
			molSmiles = Chem.MolToSmiles(mol)
			if removeSalts:
				molSmiles = removeSaltsFromSmiles(molSmiles)
			if not mol.GetProp("_Name") == "":
				molName = mol.GetProp("_Name")
			else:
				molName = molSmiles							
			output.append(f"{molName};{nConfs};{molSmiles}\n")
		return output
				
	elif fileExt == "mol":
		mol = Chem.MolFromMolFile(inputPath)
		if not mol is None:
			molSmiles = Chem.MolToSmiles(mol)
			if removeSalts:
				molSmiles = removeSaltsFromSmiles(molSmiles)
			if not mol.GetProp("_Name") == "":
				molName = mol.GetProp("_Name")
			else:
				molName = molSmiles							
			output.append(f"{molName};{nConfs};{molSmiles}\n")
		return output
		
	elif fileExt == "sdf":
		suppl = Chem.ForwardSDMolSupplier(inputPath)
		for mol in tqdm(suppl, disable=disableTqdm):
			if not mol is None:
				molSmiles = Chem.MolToSmiles(mol)
				if removeSalts:
					molSmiles = removeSaltsFromSmiles(molSmiles)
				if not mol.GetProp("_Name") == "":
					molName = mol.GetProp("_Name")
				else:
					molName = molSmiles						
				output.append(f"{molName};{nConfs};{molSmiles}\n")
			else:
				print(f"Couldn't process molecule from {inputPath}")
			
	else:
		print(f"Error with {inputPath}, file format not recognized")
		
	return output

## Code to distribute the query ligands among the amount of jobs 
def split(a, n):
    if n > len(a):
        print("more jobs than files, launching 1 job per file")
        return [a[i:i+1] for i in range(len(a))]
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

## Check if input is a file or a directory
if os.path.isfile(ligandsPath):
	print(f"Processing {ligandsPath}")
	outputLines = processInputFile(ligandsPath, disableTqdm=False, removeSalts=removeSalts)
	
elif os.path.isdir(ligandsPath):
	outputLines = []
				
	## Glob for files in target directory
	ligandPaths = sorted(list(glob.glob(f"{ligandsPath}/*.*")))
	
	for ligandPath in tqdm(ligandPaths, desc="Total Progress"):
		if os.path.isfile(ligandPath):
			outputLines += processInputFile(ligandPath, disableTqdm=True, removeSalts=removeSalts)	
else: 
	sys.exit("Error")

splitFiles = split(outputLines, nJobs)

## Writing the final csv files
for i, splitLines in enumerate(splitFiles):
	with open(f"{outputDir}/ligands_{i+1}.csv", 'w') as jobCSV:
		jobCSV.write("molName;numConfs;molSmiles\n")
		for splitLine in splitLines:
			jobCSV.write(splitLine) 
		