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

interaction_df = pd.read_csv(INTERACTION_PATH, sep=",")

# TODO: raise error if a "Source", "Target" interaction pair
# is not found in data molecules
data_molecules = set(data_df.index.to_list())
interaction_df = interaction_df[interaction_df["Source"].isin(data_molecules)]
interaction_df = interaction_df[interaction_df["Target"].isin(data_molecules)]

interaction_df = interaction_df.iloc[:1, :]

source_indexes = interaction_df["Source"]
target_indexes = interaction_df["Target"]

reference_indexes = description_df.loc[
    description_df["Group"] == REFERENCE_GROUP,
    "Sample"
].to_list()

experimental_indexes = description_df.loc[
    description_df["Group"] == EXPERIMENTAL_GROUP,
    "Sample"
].to_list()

print("Reference correlations")
ref_corrs = correlation(
    data_df[reference_indexes],
    source_indexes,
    target_indexes,
    process_num=PROCESS_NUMBER
    # numerical_index=numerical_flag
)

print("Experimental correlations")
exp_corrs = correlation(
    data_df[experimental_indexes],
    source_indexes,
    target_indexes,
    process_num=PROCESS_NUMBER
    # numerical_index=numerical_flag
)

# The hypothesis check
# print("Test phase")
# stat, pvalue = core.corr_diff_test(
#     ref_corrs.astype("float32"), np.zeros(len(ref_corrs), dtype="int32") +
#         len(description_df.loc[description_df["Group"] == REFERENCE_GROUP]),
#     exp_corrs.astype("float32"), np.zeros(len(exp_corrs), dtype="int32") +
#         len(description_df.loc[description_df["Group"] == EXPERIMENTAL_GROUP]),
#     correlation=CORRELATION,
#     alternative=ALTERNATIVE,
#     process_num=PROCESS_NUMBER
# )


# Bootstrap pvalue computation
print("Bootstrap phase")
boot_ref_corrs, boot_exp_corrs, \
boot_stat, boot_pv, boot_pvalue = \
core.corr_diff_test_boot(
    data_df,
    source_indexes,
    target_indexes,
    reference_indexes,
    experimental_indexes,
    correlation=CORRELATION,
    alternative=ALTERNATIVE,
    repeats_num=100,
    process_num=PROCESS_NUMBER
)

# adjusted_pvalue = pvalue * len(pvalue) / \
#     scipy.stats.rankdata(pvalue)
# adjusted_pvalue[adjusted_pvalue > 1] = 1
# adjusted_pvalue = adjusted_pvalue.flatten()

# Generate report
print("Report phase")
output_df = pd.DataFrame(interaction_df)
output_df["Reference"] = ref_corrs 
output_df["Experimental"] = exp_corrs 
# output_df["Statistic"] = stat
# output_df["Pvalue"] = pvalue
output_df["BReference"] = boot_ref_corrs 
output_df["BExperimental"] = boot_exp_corrs 
# output_df["BStatistic"] = boot_stat
# output_df["BPv"] = boot_pv
# output_df["BPvalue"] = boot_pvalue
# output_df["FDR"] = adjusted_pvalue
# output_df = output_df.sort_values(["FDR", "Pvalue"])

output_df.to_csv(
    OUTPUT_DIR_PATH.rstrip("/") + \
    "/{}_report.csv".format(CORRELATION),
    sep=",",
    index=None
)
