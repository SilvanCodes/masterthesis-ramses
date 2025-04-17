rule compute_gpn:
    input:
        "resources/{accession}.fna.gz",
    output:
        "results/{accession}/{chromosome}/{start_position}_{stop_position}/gpn_scores.parquet",
    conda:
        "../envs/gpn.yaml"
    resources:
        slurm_extra="-G h100:1",
    script:
        "../scripts/compute_gpn.py"
