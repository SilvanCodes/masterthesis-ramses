import pandas as pd
import bioframe as bf
from snakemake.script import snakemake

# from gtfparse import read_gtf


# gtf = read_gtf(snakemake.input[0], expand_attribute_column=False)

gtf = pd.read_csv(
    snakemake.input[0],
    sep="\t",
    header=None,
    comment="#",
    dtype={"chrom": str},
    names=[
        "chrom",
        "source",
        "feature",
        "start",
        "end",
        "score",
        "strand",
        "frame",
        "attribute",
    ],
)

# why??? in gpn.data.load_table
gtf.start -= 1

# add missing region entries
for chrom in ["chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8"]:
    start = 0
    end = gtf[gtf["chrom"] == chrom]["end"].max()
    region_entry = {
        "chrom": chrom,
        "source": ".",
        "feature": "region",
        "start": start,
        "end": end,
        "score": ".",
        "strand": "+",
        "frame": ".",
        "attribute": ".",
    }
    gtf.loc[len(gtf)] = region_entry


genic_features = [
    "gene",
    "transcript",
]

chrom_regions = gtf[gtf.feature == "region"][["chrom", "start", "end"]]

genic_intervals = gtf[gtf.feature.isin(genic_features)][["chrom", "start", "end"]]

genic_intervals = bf.merge(genic_intervals)


intergenic = bf.subtract(chrom_regions, genic_intervals)
# subtract uses end of subtracted interval as start, it seems
intergenic["start"] = intergenic["start"] + 1
intergenic["feature"] = "intergenic"

gtf = pd.concat([gtf, intergenic], ignore_index=True)


gtf_exon = gtf[gtf.feature == "exon"]
exonic_intervals = bf.merge(gtf_exon)[["chrom", "start", "end"]]
intronic_intervals = bf.subtract(genic_intervals, exonic_intervals)
# subtract uses end of subtracted interval as start, it seems
intronic_intervals["start"] = intronic_intervals["start"] + 1

intronic_intervals["feature"] = "intron"

gtf = pd.concat([gtf, intronic_intervals], ignore_index=True)


gtf_cds = gtf[gtf.feature == "CDS"]
gene_cds_overlap = bf.overlap(genic_intervals, gtf_cds)
non_coding_genes = gene_cds_overlap[gene_cds_overlap["chrom_"].isnull()][
    ["chrom", "start", "end"]
]
non_coding_genes["feature"] = "ncRNA_gene"


gtf = pd.concat([gtf, non_coding_genes], ignore_index=True)

gtf = gtf.drop_duplicates(subset=["chrom", "start", "end", "feature"])

gtf.to_parquet(snakemake.output[0], index=False)
