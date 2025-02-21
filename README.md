# Documentation of what I do on this machine

## Preamble
Added ssh-keys of desktop an laptop machines.

## Setup development environment
1. Connect to machine via Remote-SSH plugin of vscode
2. Prepare devenv via [apptainer](https://apptainer.org/docs/user/1.3/quick_start.html)
3. Load cuda module: `module load cuda/12.5`

### apptainer usage
Build .sif
```sh
apptainer build --nv devenv.sif devenv.def
```

Run .sif interactively
```sh
apptainer shell --nv devenv.sif
```

Run .sif as jupyter server in background
```sh
apptainer instance start devenv.sif devenv-instance-1
```
> available on http://localhost:8888

## Get interactive node with some GPUS

```sh
srun -p interactive --time=2:00:00 --mem=100gb -G 2 --ntasks=2 --cpus-per-task=8 --pty /bin/bash
```

Run apptainer with [`--nv` flag](https://apptainer.org/docs/user/main/gpu.html) to make cuda and graphics cards accessible.

```sh
apptainer shell --nv devenv.sif
```

Install flash-attention.
> see here: https://github.com/Dao-AILab/flash-attention

```sh
pip install flash-attn --no-build-isolation
```

Start jupyter notebook server.
```sh
jupyter notebook --no-browser --port 9999
```

Setup portforwarding of compute node via login node to local machine in new shell. (https://people.cs.umass.edu/~kexiao/posts/jupyter_notebook_remote_slurm.html)
```sh
ssh -t -t sbuedenb@ramses -L 9999:localhost:8008 ssh ramses15229 -L 8008:localhost:9999
```

## Spike: Make a NDM from GPN
> see: scripts/nucleotide_dependency_maps_for_gpn.ipynb

## Spike: Re-run UMAP plot from GPN paper

Clone gpn repository

Had to install many new python libs, see requirements.txt

Had to install vcftools

Had to fix `rename_reference` rule in Snakefile

Had to get a node with proper GPU (more than ~4GB VRAM) to run `run_umap` rule

## Workaround for ncbi-datasets-cli (autority error)

- check NCBI page of genome ([Tribolium](https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_031307605.1/))
- handpick files from FTP server

```bash
wget https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/031/307/605/GCF_031307605.1_icTriCast1.1/GCF_031307605.1_icTriCast1.1_genomic.fna.gz

wget https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/031/307/605/GCF_031307605.1_icTriCast1.1/GCF_031307605.1_icTriCast1.1_genomic.gff.gz
```


## General stuff

List default AG:
`sacctmgr show assoc -n user=$USER format=Account`

Put ninja in path in case flash-attn has to be build.

`export PATH='/home/sbuedenb/.local/bin':$PATH`


```bash
salloc -p gpu --ntasks=1 --cpus-per-task=16 --time=100:00 --mem=40gb -G 2
srun --pty bash
```
> --ntasks => "cores"
> --cpus-per-task => "threads"
> -G 1 => "GPUs"


check time left on slurm job
```bash
squeue -h -j $SLURM_JOBID -o %L
```


## Problems

```bash
salloc -p gpu --ntasks=1 --cpus-per-task=16 --time=300:00 --mem=40gb -G 1
...
47%|██████████████████████████████████████                                           | 1258/2675 [1:31:09<1:42:45,  4.35s/it]
salloc: Job 255505 has exceeded its time limit and its allocation has been revoked.
srun: forcing job termination
Hangup
[sbuedenb@ramses4 masterthesis]$ srun: Job step aborted: Waiting up to 32 seconds for job step to finish.
slurmstepd: error: *** STEP 255506.0 ON ramses16304 CANCELLED AT 2024-12-20T14:27:46 ***
srun: error: ramses16304: task 0: Killed
srun: Terminating StepId=255506.0
tcsetattr: Input/output error

---

salloc -p gpu --ntasks=1 --cpus-per-task=16 --time=300:00 --mem=40gb -G 2
...
66%|█████████████████████████████████████████████████▍                         | 882/1338 [2:43:55<1:24:51, 11.17s/it]
salloc: Job 255506 has exceeded its time limit and its allocation has been revoked.
srun: forcing job termination
Hangup
[sbuedenb@ramses4 masterthesis]$ srun: Job step aborted: Waiting up to 32 seconds for job step to finish.
slurmstepd: error: *** STEP 255532.0 ON ramses16301 CANCELLED AT 2024-12-20T17:55:18 ***
srun: error: ramses16301: task 0: Killed
srun: Terminating StepId=255532.0
tcsetattr: Input/output error

---

salloc -p gpu --ntasks=1 --cpus-per-task=16 --time=6:00:00 --mem=100gb -G 4
...
99%|█████████████████████████████████████████████████████████████████████████████▎| 663/669 [2:03:40<01:07, 11.19s/it]
salloc: Job 255532 has exceeded its time limit and its allocation has been revoked.
srun: forcing job termination
Hangup
[sbuedenb@ramses4 masterthesis]$ srun: Job step aborted: Waiting up to 32 seconds for job step to finish.
slurmstepd: error: *** STEP 255604.0 ON ramses16304 CANCELLED AT 2024-12-20T20:08:49 ***
srun: error: ramses16304: task 0: Killed
srun: Terminating StepId=255604.0
tcsetattr: Input/output error

---

salloc -p gpu --ntasks=1 --cpus-per-task=16 --time=6:00:00 --mem=100gb -G 4
...



```