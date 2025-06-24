# import pandas as pd
# import numpy as np
# from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
# from sklearn.model_selection import cross_val_predict, LeaveOneGroupOut
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from joblib import Memory

import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import cross_val_predict, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

windows = pd.read_parquet(snakemake.input[0])
features = pd.read_parquet(snakemake.input[1])


clf = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "linear",
            LogisticRegressionCV(
                random_state=42,
                verbose=True,
                max_iter=1000,
                class_weight="balanced",
                n_jobs=-1,
            ),
        ),
    ]
)
preds = cross_val_predict(
    clf,
    features,
    windows.Region,
    groups=windows.chrom,
    cv=LeaveOneGroupOut(),
    verbose=True,
)
pd.DataFrame({"pred_Region": preds}).to_parquet(output[0], index=False)

# memory = Memory(location="/scratch/sbuedenb/cache/.sk_cache", verbose=0)

# windows = pd.read_parquet(snakemake.input[0])
# features = pd.read_parquet(snakemake.input[1])

# want = [
#     # "NC_087395.1", # chr2
#     # "NC_087396.1", # chr3
#     # "NC_087397.1", # chr4
#     # "NC_087398.1", # chr5
#     # "NC_087399.1", # chr6
#     "NC_087403.1", # chr10
#     "NC_087404.1" # chr11
# ]

# # subset to selected chromosomes
# windows = windows[windows.chrom.isin(want)]
# features = features.loc[windows.index]

# X = features.to_numpy(dtype=np.float32)
# y = windows.Region.values
# groups = windows.chrom.values

# clf = Pipeline(
#     [
#         ("scaler", StandardScaler(copy=False)),
#         (
#             "linear",
#             LogisticRegressionCV(
#                 solver="saga",
#                 random_state=42,
#                 verbose=2,
#                 max_iter=500,
#                 class_weight="balanced",
#                 n_jobs=1,
#             ),
#         ),
#     ],
#     memory=memory,
# )

# preds = cross_val_predict(
#     clf,
#     X,
#     y,
#     groups=groups,
#     cv=LeaveOneGroupOut(),
#     verbose=2,
#     n_jobs=-1,
# )

# pd.DataFrame({"pred_Region": preds}).to_parquet(snakemake.output[0], index=False)
