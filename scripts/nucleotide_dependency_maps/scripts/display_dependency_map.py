import nucleotide_dependency_map_helpers as ndm
import pandas as pd

# load sequence
dependency_map_path = snakemake.input[0]

dependency_map = pd.read_parquet(dependency_map_path).values

ndm.map_to_file(dependency_map, snakemake.output[0], snakemake.wildcards.format)


# with open(snakemake.output[0], "w") as f:
#     df = pd.DataFrame({'data': np_array})
#     df.to_parquet('output.parquet')
#     f.write("Hello, world!")

# do_something(snakemake.input[0], snakemake.output[0], snakemake.threads, snakemake.config["myparam"])
