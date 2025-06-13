rule compute_gpn:
    input:
        "resources/{accession}.HMA4_region.fa",
    output:
        "results/{accession}/{chromosome}/{start_position}_{stop_position}_{reverse_complement}/gpn_scores.parquet",
    conda:
        "../envs/gpn.yaml"
    resources:
        slurm_extra="-G h100_1g.12gb:1",
    script:
        "../scripts/compute_gpn.py"
