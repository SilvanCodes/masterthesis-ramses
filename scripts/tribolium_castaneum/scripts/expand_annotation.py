import pandas as pd
import bioframe as bf
import re
from gpn.data import load_fasta, load_table
# from gtfparse import read_gtf


# gtf = read_gtf(snakemake.input[0], expand_attribute_column=False)

gtf = load_table(snakemake.input[0])

genome = load_fasta(snakemake.input[1])

# extract repeat regions from ref. genome
for chrom in gtf.chrom.unique():
    matches = list(re.finditer(r'[a-z]+', genome[chrom]))
    stretches = [(m.start(), m.end()) for m in matches]
    repeats = pd.DataFrame(
        stretches, columns=["start", "end"], dtype="int64"
    )
    repeats["chrom"] = chrom
    
    repeats["feature"] = "Repeat"
    repeats["source"] = "Derived"
    repeats["strand"] = "+"
    gtf = pd.concat([gtf, repeats], ignore_index=True)


genic_features = [
    "gene",
    ### entirely covered from gene 
    # "mRNA",
    # "CDS",
    # "ncRNA",
    # "transcript",
    # "lnc_RNA",
    # "primary_transcript",
    # "tRNA",
    # "snRNA",
    # "miRNA",
    # "rRNA",
    # "snoRNA",
    # "piRNA",
    # "cDNA_match",

    ### partially covered from gene
    "five_prime_UTR",
    "three_prime_UTR",
    "exon",
    "pseudogene",

    ### not conclusively genic
    # "Repeat"
]

chrom_regions = gtf[gtf.feature == "region"][["chrom", "start", "end"]]

genic_intervals = gtf[gtf.feature.isin(genic_features)][
    ["chrom", "start", "end"]
]

genic_intervals = bf.merge(genic_intervals)


intergenic = bf.subtract(chrom_regions, genic_intervals)
# subtract uses end of subtracted interval as start, it seems
intergenic['start'] = intergenic['start'] + 1
intergenic["feature"] = "intergenic"

gtf = pd.concat([gtf, intergenic], ignore_index=True)


gtf_exon = gtf[gtf.feature == "exon"]
exonic_intervals = bf.merge(gtf_exon)[["chrom", "start", "end"]]
intronic_intervals = bf.subtract(genic_intervals, exonic_intervals)
# subtract uses end of subtracted interval as start, it seems
intronic_intervals['start'] = intronic_intervals['start'] + 1

intronic_intervals["feature"] = "intron"

gtf = pd.concat([gtf, intronic_intervals], ignore_index=True)


gtf_cds = gtf[gtf.feature=="CDS"]
gene_cds_overlap = bf.overlap(genic_intervals, gtf_cds)
non_coding_genes = gene_cds_overlap[gene_cds_overlap['chrom_'].isnull()][["chrom", "start", "end"]]
non_coding_genes["feature"] = "ncRNA_gene"


gtf = pd.concat([gtf, non_coding_genes], ignore_index=True)

gtf = gtf.drop_duplicates(subset=["chrom", "start", "end", "feature"])

gtf.to_parquet(snakemake.output[0], index=False)