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

Setup portforwarding of compute n
ode via login node to local machine in new shell. (https://people.cs.umass.edu/~kexiao/posts/jupyter_notebook_remote_slurm.html)
```sh
ssh -t -t sbuedenb@ramses4.itcc.uni-koeln.de -L 9999:localhost:8008 ssh ramses15233 -L 8008:localhost:9999
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

## Setup without apptainer

```bash
module load lang/Miniconda3/23.9.0-0 && \
conda env create --file environment.yml && \
conda activate snakemake
```

## snakemake good to know

Build conda environments:
```
snakemake --sdm conda --conda-create-envs-only
```

Run with conda and specific profile:
```
snakemake --sdm conda --profile=profiles/slurm
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

## Train a GPN

```
export WANDB_API_KEY=
export WANDB_ENTITY=sbuedenb-university-of-cologne
export WANDB_PROJECT=beetle-gpn

export RUN_NAME=wide-cosine-1024
export OUTPUT_DIR="/scratch/sbuedenb/gpn-training/$RUN_NAME/"
```
then
```
conda activate gpn_gpu
```
then
```
torchrun --nproc_per_node=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}') -m gpn.ss.run_mlm \
    --do_train \
    --do_eval \
    --report_to wandb \
    --prediction_loss_only True \
    --remove_unused_columns False \
    --dataset_name sbuedenb/big_beetle_dataset-1024 \
    --tokenizer_name gonzalobenegas/tokenizer-dna-mlm \
    --soft_masked_loss_weight_train 0.1 \
    --soft_masked_loss_weight_evaluation 0.0 \
    --total_batch_size 2048 \
    --weight_decay 0.01 \
    --optim adamw_torch \
    --dataloader_num_workers 4 \
    --seed 42 \
    --save_strategy steps \
    --save_steps 5000 \
    --eval_strategy steps \
    --eval_steps 1000 \
    --logging_steps 100 \
    --max_steps 180000 \
    --warmup_steps 1000 \
    --learning_rate 4e-3 \
    --lr_scheduler_type cosine \
    --run_name $RUN_NAME \
    --output_dir $OUTPUT_DIR \
    --model_type GPN \
    --per_device_train_batch_size 256 \
    --per_device_eval_batch_size 256 \
    --gradient_accumulation_steps 1 \
    --ddp_find_unused_parameters False \
    --bf16 \
    --bf16_full_eval \
    --ignore_data_skip \
    --config_overrides "first_kernel_size=9,rest_kernel_size=9,dilation_max=243,dilation_cycle=6,dilation_base=3,num_hidden_layers=18"
```
    --ignore_data_skip \
    --torch_compile \
    --lr_scheduler_type constant_with_warmup \



```
    --nnodes=1:2 \
    --nproc-per-node=4 \
    --max-restarts=6 \
    --standalone \
```
### fuck up counter: 7

- space in command from repo
- python 3.12 too new (3.10 should be stable)
- imports `is_torch_tpu_available` which is deprecated without even using it
- imports `from scipy.stats import geom` without documenting or using
- dataset is not local ?
- SegFault
- random crash

###
salloc \
  --mail-user=sbuedenb@smail.uni-koeln.de \
  --mail-type=BEGIN \
  --nodes=1 \
  --ntasks=4 \
  --cpus-per-task=16 \
  --time=48:00:00 \
  --mem=64gb \
  -p gpu \
  -G h100:4
> srun --pty bash

scontrol update JobId=661168 StartTime=10:00:00

### training notes
#### "GPN" current gpn architecture (first-try:small_data, third-try-big-data) #! continue RUN_ID: 9lptl80n
Number of trainable parameters = 93.091.328
--per_device_train_batch_size 256 \
--per_device_eval_batch_size 256 \

third-try-big-data: v1 is 240.000

#### "GPN" current gpn architecture with dropout (second-try)
Number of trainable parameters = 93.091.328
--config_overrides "hidden_dropout_prob=0.1"
--per_device_train_batch_size 256 \
--per_device_eval_batch_size 256 \

#### "GPN" with wide flat pyramid kernel config (wide_flat_pyramid_kernel_k9b3:small_data, wide-net-big-data)
Number of trainable parameters = 118.257.152
--config_overrides "first_kernel_size=9,rest_kernel_size=9,dilation_max=81,dilation_cycle=5,dilation_base=3"
--per_device_train_batch_size 256 \
--per_device_eval_batch_size 256 \

#### "GPN" with wide flat pyramid kernel config (long-wide:long-data) #! fresh start
--dataset_name sbuedenb/big_beetle_dataset-2048 \
--config_overrides "first_kernel_size=9,rest_kernel_size=9,dilation_max=243,dilation_cycle=6,dilation_base=3,num_hidden_layers=18"
--per_device_train_batch_size 128 \
--per_device_eval_batch_size 128 \
--learning_rate 2e-3 \
--lr_scheduler_type cosine \
--max_steps 180000

Total batch size: 4 GPU * 128/GPU = 512 with 2048 tokens per sample => 1.048.576 tokens per batch
Number of trainable parameters = 85,219,840

Try: --learning_rate 2e-3 \
     --lr_scheduler_type cosine \
     --

#### "GPN" less deep with wide flat pyramid kernel config (wide-big-reduced) #! continue RUN_ID: g5dyoby9
--config_overrides "first_kernel_size=9,rest_kernel_size=9,dilation_max=81,dilation_cycle=5,dilation_base=3,num_hidden_layers=20"
--per_device_train_batch_size 256 \
--per_device_eval_batch_size 256 \

Total batch size: 4 GPU * 256/GPU = 1024 with 512 tokens per sample => 524.288 tokens ber batch
Number of trainable parameters = 94.659.072

wide-big-reduced: v1 is 120.000, v2 is 240.000


torchrun --nproc_per_node=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}') -m gpn.ss.run_mlm \
    --do_train \
    --do_eval \
    --report_to wandb \
    --prediction_loss_only True \
    --remove_unused_columns False \
    --dataset_name sbuedenb/big_beetle_dataset \
    --tokenizer_name gonzalobenegas/tokenizer-dna-mlm \
    --soft_masked_loss_weight_train 0.1 \
    --soft_masked_loss_weight_evaluation 0.0 \
    --total_batch_size 2048 \
    --weight_decay 0.01 \
    --optim adamw_torch \
    --dataloader_num_workers 9 \
    --seed 42 \
    --save_strategy steps \
    --save_steps 5000 \
    --eval_strategy steps \
    --eval_steps 1000 \
    --logging_steps 100 \
    --num_train_epochs 26 \
    --learning_rate 1e-3 \
    --lr_scheduler_type cosine_with_restarts \
    --lr_scheduler_kwargs '{"num_cycles":3}' \
    --max_steps 114169 \
    --warmup_steps 1000 \
    --load_best_model_at_end True \
    --metric_for_best_model loss \
    --run_name $RUN_NAME \
    --output_dir $OUTPUT_DIR \
    --model_type GPN \
    --per_device_train_batch_size 256 \
    --per_device_eval_batch_size 256 \
    --gradient_accumulation_steps 1 \
    --ddp_find_unused_parameters False \
    --bf16 \
    --bf16_full_eval \
    --ignore_data_skip \
    --config_overrides "first_kernel_size=9,rest_kernel_size=9,dilation_max=81,dilation_cycle=5,dilation_base=3,num_hidden_layers=20"

--lr_scheduler_type cosine
--max_steps 30000
--resume_from_checkpoint <model_path>


#### "GPN" smol
--config_overrides "first_kernel_size=9,rest_kernel_size=5,dilation_max=64,dilation_cycle=7,dilation_base=2,num_hidden_layers=21,hidden_size=256,intermediate_size=1024,hidden_dropout_prob=0.1"
Number of trainable parameters = 19.608.320

#### "ConvNet" gpn legacy
Number of trainable parameters = 65880071

### resume WANDB run
```
export WANDB_RESUME=allow
export WANDB_RUN_ID="9inaq395"
```

> RUN_ID must be explicit id not name

### upload to huggingface

> put stuff into new folder `v2`

huggingface-cli upload sbuedenb/beetle-gpn ./v2 . --commit-message "v2 – retrained with dropout 0.1"


https://www.ncbi.nlm.nih.gov/datasets/docs/v2/policies-annotation/genomeftp/#are-repetitive-sequences-in-eukaryotic-genomes-masked



