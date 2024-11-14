# Documentation of what I do on this machine

## Preamble
Added ssh-keys of desktop an laptop machines.

## Setup development environment
1. Connect to machine via Remote-SSH plugin of vscode
2. Prepare devenv via [apptainer](https://apptainer.org/docs/user/1.3/quick_start.html)

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


## General stuff

List default AG:
`sacctmgr show assoc -n user=$USER format=Account`