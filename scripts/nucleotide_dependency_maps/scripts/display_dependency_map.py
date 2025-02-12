import nucleotide_dependency_map_helpers as ndm
import pandas as pd

from gpn.data import load_fasta

chromosome = snakemake.config["CHROMOSOME"]
seq_start = snakemake.config["START_POSITION"]
seq_end = snakemake.config["END_POSITION"]

map_start = snakemake.config["DISPLAY_MAP_START_RELATIVE"]
map_end = snakemake.config["DISPLAY_MAP_END_RELATIVE"]


# load sequence
sequence_path = snakemake.input[0]
genome = load_fasta(sequence_path)
sequence = genome[chromosome][seq_start - 1 : seq_end]

# load dep_map
dependency_map_path = snakemake.input[1]
dependency_map = pd.read_parquet(dependency_map_path).values

print(dependency_map.shape)

title = f"{chromosome} {seq_start + map_start} - {map_end + seq_end + 1}"

ndm.map_seq_to_file(
    dependency_map[map_start:map_end, map_start:map_end],
    sequence[map_start:map_end],
    snakemake.output[0],
    snakemake.wildcards.format,
    vmax=snakemake.config["DISPLAY_VMAX"],
    title=title,
)
