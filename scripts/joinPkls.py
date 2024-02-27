#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 20:42:36 2024

@author: Jochem Nelen (jnelen@ucam.edu)
"""

## A script to join the output pkls into one (compressed) pkl. Addtionally gives an overview of the runtimes and failed compounds
## Usage: Call this script (using the singularity image located at singularity/TorsionalDiffusionHPC.sif) and the output directory of a TorsionalDiffusionHPC run

import glob
import gzip
import os
import pickle
import rdkit
import sys

from statistics import mean, stdev

args = sys.argv

inputDir = args[1]

if len(args) > 2:
	outputPath = args[2]
else:
	outputPath = inputDir + "/joinedMolecules.pkl.gz"	

outputDict = {}

runTimeList = []
failList = []

pklPaths = glob.glob(inputDir + "/molecules/*.pkl*")

pklPaths = sorted(pklPaths, key=lambda x: int(x.split('.')[0].split('_')[-1]))

counter = 0
totalCompounds = 0

for pklPath in pklPaths:

	jobNumber = pklPath.split("_")[-1].split(".")[0]
	jobOutFiles = glob.glob(f"{inputDir}/jobs_out/job_{jobNumber}_*.out")
	
	if pklPath.split('.')[-1].lower() == "gz":
		pklPath = gzip.open(pklPath, 'rb')

	try:
		molDict = pickle.load(pklPath)
	except:
		print(f"Couldn't process {pklPath}")
		continue
		
	if type(molDict) is dict:
		outputDict.update(molDict)
		counter += 1

	runTime = None
	
	for jobOutFile in jobOutFiles:
		with open(jobOutFile) as outFile:
			outLines = outFile.readlines()

			## Check if this output file correctly finished running
			for outLine in outLines:
				if "Finished after " in outLine:
					runTime = float(outLine.split("Finished after")[-1].split(" seconds")[0])
					break

			if not runTime == None:
				runTimeList.append(runTime)
				
				## Loop again through the lines to get the molnames of all compounds that gave errors
				for outLine in outLines:

					if " with " in outLine:
						failList.append(outLine.split(" with ")[0].strip().split(" ")[-1])
						
						continue
					elif "Starting TorsionalDiffusion Calculations for " in outLine:
						totalCompounds += int(outLine.split("Starting TorsionalDiffusion Calculations for ")[-1].split(" compounds..")[0])
						continue
				break
	
if ".gz" in outputPath:
	f = gzip.open(outputPath, 'wb')
else:
	f = open(outputPath, 'wb')	

pickle.dump(outputDict, f)
f.close()

print(f"Joined {counter} pkl files to {outputPath}")

if os.path.exists(inputDir + "/energies/"):
	
	energyPaths = sorted(glob.glob(inputDir + "/energies/*.csv"), key=lambda x: int(x.split('.')[0].split('_')[-1]))
	
	energyOutputPath = outputPath.split(".")[0] + ".csv"
	with open(energyOutputPath, "w") as energyFile:
		energyFile.write("mol_name;rot_bonds;n_confs;rmsd;F;energy;dlogp;smiles\n")
		for energyPath in energyPaths:
			with open(energyPath) as energyCsv:
				energyLines = energyCsv.readlines()
				for energyLine in energyLines[1:]:
					energyFile.write(energyLine) 
	print(f"Joined energy csvs to {energyOutputPath}")
print(f"-------Jobs summary-------")
print(f"Average Runtime: {mean(runTimeList):.2f} +/- {stdev(runTimeList):.2f} seconds")
print(f"{len(outputDict.keys())} compounds succeeded, while {len(failList)} out of {totalCompounds} compounds ({(len(failList)/totalCompounds)*100:.2f}%) failed:")
print(failList)

with open(f"{inputDir}/summary.txt", "w") as outputFile:
	outputFile.write(f"-------Jobs summary-------\n")
	outputFile.write(f"Work Directory: {inputDir}\n")
	outputFile.write(f"Average Runtime: {mean(runTimeList):.2f} +/- {stdev(runTimeList):.2f} seconds\n")
	outputFile.write(f"{len(outputDict.keys())} compounds succeeded, while {len(failList)} out of {totalCompounds} compounds ({(len(failList)/totalCompounds)*100:.2f}%) failed:\n")
	outputFile.write(str(failList))
 
