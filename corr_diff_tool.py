import numpy as np
import pandas as pd
import scipy.stats

import sys
import time
import tqdm

# Import python package
import core
from config import *


data_df = pd.read_csv(DATA_PATH, sep=",", index_col=0)
description_df = pd.read_csv(DESCRIPTION_PATH, sep=",")

if (CORRELATION == "spearman"):
    correlation = core.spearmanr
else:
    corr_test = core.pearsonr

interaction_df = None
if (INTERACTION_PATH):
    interaction_df = pd.read_csv(INTERACTION_PATH, sep=",", index_col=0)
    source_indexes = interaction_df["Source"]
    taget_indexes = interaction_df["Target"]

else:
    index_arr = np.array(np.meshgrid(
        np.arange(len(data_df)),
        np.arange(len(data_df))
    )).T.reshape(-1, 2)
    
    index_arr = index_arr[:(len(index_arr) + 1) // 2, :]
    # index_arr = index_arr[:10 // 2, :]
    
    source_indexes = index_arr[:, 0]
    target_indexes = index_arr[:, 1]
    
    # data_indexes = np.array(data_df.index.to_list())
    # source_indexes = data_indexes[source_num_indexes]
    # target_indexes = data_indexes[target_num_indexes]
    
    # interaction_df = pd.DataFrame(source_indexes, columns=["Source"])
    # interaction_df["Target"] = target_indexes

numerical_flag = True
if INTERACTION_PATH:
    numerical_flag = False

print("Reference correlations")
ref_corrs = correlation(
    data_df[
        description_df.loc[
            description_df["Group"] == REFERENCE_GROUP,
            "Sample"
        ].to_list()
    ],
    source_indexes,
    target_indexes,
    numerical_index=numerical_flag
)

print("Experimental correlations")
exp_corrs = correlation(
    data_df[
        description_df.loc[
            description_df["Group"] == EXPERIMENTAL_GROUP,
            "Sample"
        ].to_list()
    ],
    source_indexes,
    target_indexes,
    numerical_index=numerical_flag
)

stat, pvalue = core.corr_diff_test(
    ref_corrs, len(description_df.loc[description_df["Group"] == REFERENCE_GROUP]),
    exp_corrs, len(description_df.loc[description_df["Group"] == EXPERIMENTAL_GROUP]),
    correlation=CORRELATION
)

adjusted_pvalue = pvalue * len(pvalue) / \
    scipy.stats.rankdata(pvalue)
adjusted_pvalue[adjusted_pvalue > 1] = 1

# Generate report
if INTERACTION_PATH:
    output_df = pd.DataFrame(interaction_df)
else:
    output_df = pd.DataFrame()

output_df["Reference"] = ref_corrs 
output_df["Experimental"] = exp_corrs 
output_df["Statistic"] = stat
output_df["Pvalue"] = pvalue
output_df["FDR"] = adjusted_pvalue
output_df.to_csv(OUTPUT_DIR_PATH.rstrip("/") + "/report.csv", sep=",", index=None)
