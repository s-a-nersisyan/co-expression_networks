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

# Test mode
# data_df = data_df.iloc[:100]
# End of Test mode

if (CORRELATION == "spearman"):
    correlation = core.spearmanr
else:
    correlation = core.pearsonr

interaction_df = None
if (INTERACTION_PATH):
    interaction_df = pd.read_csv(INTERACTION_PATH, sep=",", index_col=0)
    source_indexes = interaction_df["Source"]
    target_indexes = interaction_df["Target"]

else:
    source_indexes=None
    target_indexes=None

# numerical_flag = True
# if INTERACTION_PATH:
#     numerical_flag = False

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
    process_num=PROCESS_NUMBER
    # numerical_index=numerical_flag
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
    process_num=PROCESS_NUMBER
    # numerical_index=numerical_flag
)

print("Test phase", len(ref_corrs))
stat, pvalue = core.corr_diff_test(
    ref_corrs.astype("float32"), np.zeros(len(ref_corrs), dtype="int32") +
        len(description_df.loc[description_df["Group"] == REFERENCE_GROUP]),
    exp_corrs.astype("float32"), np.zeros(len(exp_corrs), dtype="int32") +
        len(description_df.loc[description_df["Group"] == EXPERIMENTAL_GROUP]),
    correlation=CORRELATION,
    alternative=ALTERNATIVE,
    process_num=PROCESS_NUMBER
)

# Test mode
# print("PyTest phase", len(ref_corrs))
# py_stat, py_pvalue = core.py_corr_diff_test(
#     ref_corrs.astype("float32"), np.zeros(len(ref_corrs), dtype="int32") +
#         len(description_df.loc[description_df["Group"] == REFERENCE_GROUP]),
#     exp_corrs.astype("float32"), np.zeros(len(exp_corrs), dtype="int32") +
#         len(description_df.loc[description_df["Group"] == EXPERIMENTAL_GROUP]),
#     correlation=CORRELATION,
#     alternative=ALTERNATIVE,
#     # process_num=PROCESS_NUMBER
# )
# End of test mode

adjusted_pvalue = pvalue * len(pvalue) / \
    scipy.stats.rankdata(pvalue)
adjusted_pvalue[adjusted_pvalue > 1] = 1

# Generate report
print("Report phase")
if INTERACTION_PATH:
    output_df = pd.DataFrame(interaction_df)

else:
    indexes = np.where(adjusted_pvalue < FDR_THRESHOLD)
    # indexes = np.arange(len(adjusted_pvalue), dtype="int32")
    ref_corrs = ref_corrs[indexes]
    exp_corrs = exp_corrs[indexes]
    stat = stat[indexes]
    pvalue = pvalue[indexes]
    adjusted_pvalue = adjusted_pvalue[indexes]
    # py_stat = py_stat[indexes]
    # py_pvalue = py_pvalue[indexes]
    df_indexes = data_df.index.to_numpy()
    source_indexes = []
    target_indexes = []
    for ind in indexes:
        s, t = core.paired_index(ind, len(df_indexes))
        source_indexes.append(df_indexes[s])
        target_indexes.append(df_indexes[t])

    output_df = pd.DataFrame()
    output_df["Source"] = source_indexes
    output_df["Target"] = target_indexes

output_df["Reference"] = ref_corrs 
output_df["Experimental"] = exp_corrs 
output_df["Statistic"] = stat
output_df["Pvalue"] = pvalue
# output_df["PyStatistic"] = py_stat
# output_df["PyPvalue"] = py_pvalue
output_df["FDR"] = adjusted_pvalue
output_df.to_csv(OUTPUT_DIR_PATH.rstrip("/") + "/{}_report.csv".format(CORRELATION), sep=",", index=None)
