# _Arabidopsis halleri_

This workflow is here in order to compute a UMAP of embeddings of the Lan3.1 genome from _Arabidopsis halleri_.

It is adopted from https://github.com/songlab-cal/gpn/blob/main/analysis/gpn_arabidopsis/Snakefile

## Preparation

This workflow needs the two files `Lan3.1.fna.gz` and `Lan3.1.genomic.gff3.gz` in a folder named `resources`.
The first file is the genome, the ssecond fiel is the annotation.
The can be downloaded here for example: https://phytozome-next.jgi.doe.gov/info/Ahalleri_v2_1_0

## Required programs

`Conda` and `Snakemake` need to be installed.

## Addons

If you have SLURM available, install `snakemake-executor-plugin-slurm` and adapt `profiles/slurm`.