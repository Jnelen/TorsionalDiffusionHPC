#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 21:48:21 2024

@author: Jochem Nelen (jnelen@ucam.edu)
"""

### Sometimes jobs fail to launch or finish. This script checks which jobs didn't produce an output and tries to relaunch them

import glob
import sys
import subprocess

inputPath = sys.argv[1]

jobList = []
finishedList = []

for path in glob.glob(f"{inputPath}/input_csvs/*.csv"):
	jobNumber = path.split("_")[-1].split(".")[0]
	jobList.append(int(jobNumber))

for path in glob.glob(f"{inputPath}/molecules/*.pkl*"):
	jobNumber = path.split("_")[-1].split(".")[0]
	finishedList.append(int(jobNumber))

finalList = [x for x in jobList if x not in finishedList]

print(f"Found {len(finalList)} jobs that didn't produce an output.")

for jobNumber in finalList:
	print(f"Relaunching job {jobNumber}")
	subprocess.run(f"sh {inputPath}/jobs/job_{jobNumber}.sh", shell=True)
