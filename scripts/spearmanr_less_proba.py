import numpy as np
import pandas as pd

import tqdm

import sys

from core.correlation import *

normal = pd.read_csv(sys.argv[1], sep=",", index_col=0)
tumored = pd.read_csv(sys.argv[2], sep=",", index_col=0)

# if (len(sys.argv) > 3):
#     graph_df = pd.rad_csv(sys.argv[3], sep=",")

TOP_SIZE = 10**4
GENE = "AR"
CORR_TRESHOLD = 0.1
PV_TRESHOLD = 0.05

top_expressed_genes = np.array(tumored.median(axis=1).sort_values(
    ascending=False
).index[:TOP_SIZE])

tumored_target = tumored.loc[GENE]
tumored = tumored.loc[top_expressed_genes].reindex(top_expressed_genes)
normal_target = normal.loc[GENE]
normal = normal.loc[top_expressed_genes].reindex(top_expressed_genes)

tumored_corrs = np.array([
    spearmanr_stats(tumored_target, sample)\
    for i, sample in tqdm.tqdm(tumored.iterrows())
])

normal_corrs = np.array([
    spearmanr_stats(normal_target, sample)\
    for i, sample in tqdm.tqdm(normal.iterrows())
])

print("Analytic computations")
pvalue = correlation_diff_proba(
    normal_target, normal,
    tumored_target, tumored,
    [0] * len(tumored_corrs),
    correlation="spearman",
    method="analytic",
    alternative="less"
)

np.save("../data/AR_analytic_pvalue.npy", pvalue)
indexes = np.argsort(-pvalue)[:10]
print(top_expressed_genes[indexes])
print(normal_corrs[indexes])
print(tumored_corrs[indexes])
print(pvalue[indexes])

print("Bootstrap computations")
pvalue = correlation_diff_proba(
    normal_target, normal.iloc[indexes],
    tumored_target, tumored.iloc[indexes],
    [0] * len(indexes),
    correlation="spearman",
    method="bootstrap",
    alternative="less"
)

np.save("../data/AR_bootstrap_pvalue.npy", pvalue)
print(top_expressed_genes[indexes])
print(normal_corrs[indexes])
print(tumored_corrs[indexes])
print(pvalue)


