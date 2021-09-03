import numpy as np
import pandas as pd
import scipy.stats

import sys
import time
import tqdm
import json

# Import python package
import core

# Arg parser
import argparse


# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("config_path")
args = parser.parse_args()

# Load config file
CONFIG_PATH = args.config_path
config = json.load(open(CONFIG_PATH, "r"))

DATA_PATH = config["data_path"]
DESCRIPTION_PATH = config["description_path"]
INTERACTION_PATH = config["interaction_path"]
OUTPUT_DIR_PATH = config["output_dir_path"]

REFERENCE_GROUP = config["reference_group"]
EXPERIMENTAL_GROUP = config["experimnetal_group"]

CORRELATION = config["correlation"]
ALTERNATIVE = config["alternative"]
PROCESS_NUMBER = config["process_number"]

FDR_THRESHOLD = config["fdr_treshold"]

# Main part
data_df = pd.read_csv(DATA_PATH, sep=",", index_col=0)
description_df = pd.read_csv(DESCRIPTION_PATH, sep=",")

if (CORRELATION == "spearman"):
    correlation = core.spearmanr
else:
    correlation = core.pearsonr

interaction_df = None
if (INTERACTION_PATH):
    interaction_df = pd.read_csv(INTERACTION_PATH, sep=",")
    
    # TODO: raise error if a "Source", "Target" interaction pair
    # is not found in data molecules
    data_molecules = set(data_df.index.to_list())
    interaction_df = interaction_df[interaction_df["Source"].isin(data_molecules)]
    interaction_df = interaction_df[interaction_df["Target"].isin(data_molecules)]
    
    source_indexes = interaction_df["Source"]
    target_indexes = interaction_df["Target"]
    
else:
    source_indexes=None
    target_indexes=None

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
adjusted_pvalue = adjusted_pvalue.flatten()

# Generate report
print("Report phase")
if INTERACTION_PATH:
    output_df = pd.DataFrame(interaction_df)

else:
    indexes = np.where(adjusted_pvalue < FDR_THRESHOLD)[0]
    ref_corrs = ref_corrs[indexes]
    exp_corrs = exp_corrs[indexes]
    stat = stat[indexes]
    pvalue = pvalue[indexes]
    adjusted_pvalue = adjusted_pvalue[indexes]
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
output_df["FDR"] = adjusted_pvalue
output_df = output_df.sort_values(["FDR", "Pvalue"])

output_df = output_df[output_df["Pvalue"] < FDR_THRESHOLD]
output_df.to_csv(
    OUTPUT_DIR_PATH.rstrip("/") + \
    "/{}_report.csv".format(CORRELATION),
    sep=",",
    index=None
)
