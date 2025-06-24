import pandas as pd
from umap import UMAP
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

embeddings = pd.read_parquet(snakemake.input[0])
proj = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("umap", UMAP(metric='cosine', random_state=42, verbose=True)),
    ]
).fit_transform(embeddings)
proj = pd.DataFrame(proj, columns=["UMAP1", "UMAP2"])
proj.to_parquet(snakemake.output[0], index=False)