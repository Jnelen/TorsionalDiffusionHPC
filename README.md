# TorsionalDiffusionHPC

TorsionalDiffusionHPC is a fork of [torsional-diffusion](https://github.com/gcorso/torsional-diffusion), which adds support to run it on HPC systems using Slurm and Singularity.  
For more details about torsional-diffusion we refer to the original [Github](https://github.com/gcorso/torsional-diffusion) repo and the [paper on arXiv](https://arxiv.org/abs/2206.01729).

## Requirements:
* Singularity 
* Slurm

## Installation instructions:
1. Clone the repository and navigate to it
    ```
    git clone https://github.com/Jnelen/TorsionalDiffusionHPC
    ```
   ```
   cd TorsionalDiffusionHPC
   ```
2. Download the singularity image (~4 GB) to the singularity directory located in the main TorsionalDiffusionHPC directory. The singularity image contains all the necessary packages and dependencies to run DiffDock correctly

   ```
   wget --no-check-certificate -r "https://drive.usercontent.google.com/download?id=1Uzx7OqghIqSoNBpZ1_2V76sMvl7XXOS2&confirm=t" -O singularity/TorsionalDiffusionHPC.sif
   ```
   
   alternatively, you can build the singularity image yourself using:
   ```
   singularity build singularity/TorsionalDiffusionHPC.sif singularity/TorsionalDiffusionHPC.def
   ```
3. Download one of the trained models to the `workdir` directory from [this shared Drive](https://drive.google.com/drive/folders/1BBRpaAvvS2hTrH81mAE4WvyLIKMyhwN7?usp=sharing). I set the default model to use [drugs_default](https://drive.google.com/drive/folders/1aW-FRtriTUpsOBy1vF495BsX4zktltg6?usp=drive_link), so I recommend installing this one to the `workdir` directory, however other models are supported as well.
    make the `workdir` to download the model to:
    ```
    mkdir workdir
    ```
    download the drugs_default model:
   ```
    wget --no-check-certificate -r "https://drive.usercontent.google.com/download?id=1Yez3v0H8trS4jAnrn8vdzt-R7TkM1L_U&confirm=t" -O workdir/drugs_default.zip
   ```
   unzip the model and remove the zip file:
    ```
    unzip workdir/drugs_default.zip -d workdir/
   ```
     ```
    rm workdir/drugs_default.zip
   ```
4. Run a test example to generate the required (hidden) .npy files. This only needs to happen once and should only take about 5-10 minutes.   
   ```
   mkdir output
   ```  
   ```
   python launch_jobs.py -l data/test.csv -out output/test -j 1
   ```  
## Options
I attempted to provide most of the original options implemented in [torsional-diffusion](https://github.com/gcorso/torsional-diffusion), while also keeping things simple.
Additionally, I added some useful features (for example compressing the results, removing salts, ...) and scripts which can make general usage easier. Here is a short overview:
### Command arguments

#### (Most relevant) TorsionalDiffusion Options

- `--ligands LIGANDS, -l LIGANDS`: The path to and sdf file or a directory of mol(2)/sdf ligand files. Csv and pkl files are also accepted as input. All of these formats are also allowed to have been compressed by gzip (.gz)
- `--out_dir OUT_DIR, -out OUT_DIR, -o OUT_DIR`: Directory where the output structures will be saved to
- `--num_confs NUM_CONFS, -n NUM_CONFS`: How many conformers to output per compound. The default value is 10
- `--dump_pymol`: Save .pdb file with denoising dynamics
- `--pre-mmff`: Run MMFF on the local structure conformer
- `--post-mmff`: Run MMFF on the final generated structures
- `--no_energy, -ne`: Skip Calculating the energies and other metrics
- `--particle_guidance {0,1,2,3,4}, -pg {0,1,2,3,4}`: Define which type of Particle Guidance you want to use:
    - 0: No particle guidance
    - 1: Permutation invariant minimize recall error
    - 2: Permutation invariant minimize precision error
    - 3: Non-permutation invariant minimize recall error
    - 4: Non-permutation invariant minimize precision error

#### Extra Options

- `--smiles_as_id, -si`: Use a molecule's smile as the ID, even if the molecule has a name
- `--compress_output, -co`: Compress the output pkl files using gzip
- `--remove_salts, -rs`: Remove salts and fragments from the input molecules
- `--random_coords, -rc`: Use the "useRandomCoords=True" option when generating initial RDKit conformers (more robust, but slower)
- `--random_seed RANDOM_SEED, --seed RANDOM_SEED`: Random seed to produce (approximate) deterministic results for identical datasets.

#### Slurm Options

- `--jobs JOBS, -j JOBS`: Number of jobs to use
- `--time TIME, -t TIME, -tj TIME`: Max amount of time each job can run
- `--queue QUEUE, -qu QUEUE`: On which node to launch the jobs. The default value is the default queue for the user. Might need to be specified if there is no default queue configured
- `--mem MEM, -m MEM`: How much memory to use for each job. The default value is 4GB
- `--gpu, -gpu, -GPU, --GPU`: Use GPU resources. This will accelerate docking calculations if a compatible GPU is available
- `--cores CORES, -c CORES`: How many cores to use for each job. The default value is 1

### Scripts
The additional scripts are located in the `scripts/` directory. Currently there are two:
- relaunchFailedJobs.py  
Sometimes jobs fail or produce errors. This can especially be annoying when running a large amount of jobs. After all jobs stopped running, but not all jobs finished successfully, you run this script to automatically rerun the jobs that didn't produce a final output.  
Usage: `python scripts/relaunchFailedJobs.py <output_directory>`
- joinPkls.py  
This script can join all the results from every job back together into one large (compressed) pkl. Additionally, energy csvs will also be joined if they were generated.  
Usage: `singularity run singularity/TorsionalDiffusionHPC.sif python scripts/joinPkls.py <output_directory>`


## License
MIT

