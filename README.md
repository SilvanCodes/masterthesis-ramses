# Documentation of what I do on this machine

## Preamble
Added ssh-keys of desktop an laptop machines.

## Setup development environment
1. Connect to machine via Remote-SSH plugin of vscode
2. Prepare devenv via [apptainer](https://apptainer.org/docs/user/1.3/quick_start.html)
3. Load cuda, etc.. module: `module load nvhpc/24.9`

### apptainer usage
Build .sif
```
apptainer build devenv.sif devenv.def
```

Run .sif interactively
```
apptainer shell devenv.sif
```

Run .sif as jupyter server in background
```
apptainer instance start devenv.sif devenv-instance-1
```
> available on http://localhost:8888

## Get interactive node with some GPUS

`srun -p interactive --time=30:00 --pty /bin/bash`

Mount folder containg modules

`apptainer shell --bind /projects devenv.sif`

Symlink or put path of (nvhpc) binaries in $PATH

`..??..`

Goal: get flash-attention to install successfully.
> see here: https://github.com/Dao-AILab/flash-attention
`pip install flash-attn --no-build-isolation`

## General stuff

List default AG:
`sacctmgr show assoc -n user=$USER format=Account`