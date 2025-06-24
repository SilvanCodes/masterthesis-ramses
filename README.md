# Welcome to the codebase behind the masterthesis "Analysis on GPN and its application to distant species"

Most happens in `scripts` and `experiments`.



`scripts/data_preparation` is pretty much a copy of https://github.com/songlab-cal/gpn/tree/main/workflow/make_dataset with different species as input.


`scripts/arabidopsis_halleri` and `scripts/tribolium_castaneum` are adaptations of https://github.com/songlab-cal/gpn/blob/main/analysis/gpn_arabidopsis/Snakefile to the respective species.


`scripts/nucleotide_dependency_maps` is an adaptation to a Snakemake workflow from https://github.com/gagneurlab/dependencies_DNALM/blob/main/compute_and_visualize_dep_maps.ipynb.


`scripts/high_throughput_gpn_computation` is an original pipeline to quickly compute gpn scores for genomes.


`experiments/context_length_prediction_impact` is the code for the analysis on utilized context size of the GPN.


`NOTES.md` is kind of of a dev-log and also contains the command used to start the training of new GPN models.


https://huggingface.co/sbuedenb has all published models and datasets.

https://api.wandb.ai/links/sbuedenb-university-of-cologne/wp5omxcc has all training and evaluation graphs of all the total 26 runs.