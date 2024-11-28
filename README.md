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
apptainer shell devenv.sif
```

Run .sif as jupyter server in background
```sh
apptainer instance start devenv.sif devenv-instance-1
```
> available on http://localhost:8888

## Get interactive node with some GPUS

```sh
srun -p interactive --time=60:00 --pty /bin/bash
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

Setup portforwarding of compute node via login node to local machine in new shell.
```sh
ssh -t -t sbuedenb@ramses -L 9999:localhost:8008 ssh ramses15229 -L 8008:localhost:9999
```

## General stuff

List default AG:
`sacctmgr show assoc -n user=$USER format=Account`

Put ninja in path in case flash-attn has to be build.

`export PATH='/home/sbuedenb/.local/bin':$PATH`
