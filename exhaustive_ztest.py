import numpy as np
import pandas as pd
import scipy.stats

import sys
import time
import tqdm
import json

# Import python package
import core.extern

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
EXPERIMENTAL_GROUP = config["experimental_group"]

CORRELATION = config["correlation"]
ALTERNATIVE = config["alternative"]
PROCESS_NUMBER = config["process_number"]

FDR_THRESHOLD = config["fdr_treshold"]


# Main part
data_df = pd.read_csv(DATA_PATH, sep=",", index_col=0)
description_df = pd.read_csv(DESCRIPTION_PATH, sep=",")

if (CORRELATION == "spearman"):
    correlation = core.extern.spearmanr
elif (CORRELATION == "pearson"):
    correlation = core.extern.pearsonr
elif (CORRELATION == "spearman_test"):
    CORRELATION = "spearman"
    correlation = core.extern.spearmanr_test

interaction_df=None
source_indexes=None
target_indexes=None

reference_indexes = description_df.loc[
    description_df["Group"] == REFERENCE_GROUP,
    "Sample"
].to_list()
experimental_indexes = description_df.loc[
    description_df["Group"] == EXPERIMENTAL_GROUP,
    "Sample"
].to_list()

# Test mode
data_df = data_df.iloc[:2]

print("Reference correlations")
ref_corrs, ref_pvalues = correlation(
    data_df[reference_indexes],
    source_indexes,
    target_indexes,
    alternative="two-sided",
    process_num=PROCESS_NUMBER
)

print("Experimental correlations")
exp_corrs, exp_pvalues = correlation(
    data_df[experimental_indexes],
    source_indexes,
    target_indexes,
    alternative="two-sided",
    process_num=PROCESS_NUMBER
)

print("Test phase")
stat, pvalue = core.extern.ztest(
    ref_corrs.astype("float32"),
    len(reference_indexes),
    exp_corrs.astype("float32"),
    len(experimental_indexes)
    correlation=CORRELATION,
    alternative=ALTERNATIVE,
    process_num=PROCESS_NUMBER
)

adjusted_pvalue = pvalue * len(pvalue) / \
    scipy.stats.rankdata(pvalue)
adjusted_pvalue[adjusted_pvalue > 1] = 1
adjusted_pvalue = adjusted_pvalue.flatten()

indexes = np.arange(len(adjusted_pvalue))

ref_corrs = ref_corrs[indexes]
ref_pvalues = ref_pvalues[indexes]

exp_corrs = exp_corrs[indexes]
exp_pvalues = exp_pvalues[indexes]

stat = stat[indexes]
pvalue = pvalue[indexes]
adjusted_pvalue = adjusted_pvalue[indexes]
df_indexes = data_df.index.to_numpy()

np.save(OUTPUT_DIR_PATH.rstrip("/") + \
    "/{}_ztest_stat.npy".format(CORRELATION),
    stat
)
np.save(OUTPUT_DIR_PATH.rstrip("/") + \
    "/{}_ztest_pvalue.npy".format(CORRELATION),
    pvalue
)
np.save(OUTPUT_DIR_PATH.rstrip("/") + \
    "/{}_ztest_fdr.npy".format(CORRELATION),
    adjusted_pvalue
)

# Test mode
# indexes = np.where(adjusted_pvalue < FDR_THRESHOLD)[0]

ref_corrs = ref_corrs[indexes]
ref_pvalues = ref_pvalues[indexes]

exp_corrs = exp_corrs[indexes]
exp_pvalues = exp_pvalues[indexes]

stat = stat[indexes]
pvalue = pvalue[indexes]
adjusted_pvalue = adjusted_pvalue[indexes]
df_indexes = data_df.index.to_numpy()

source_indexes = []
target_indexes = []
for ind in tqdm.tqdm(indexes):
    s, t = core.extern.paired_index(ind, len(df_indexes))
    source_indexes.append(df_indexes[s])
    target_indexes.append(df_indexes[t])

source_indexes = np.array(source_indexes, dtype=np.str)
target_indexes = np.array(target_indexes, dtype=np.str)

output_df = pd.DataFrame()
output_df["Source"] = source_indexes
output_df["Target"] = target_indexes

output_df["RefCorr"] = ref_corrs 
output_df["RefPvalue"] = ref_pvalues

output_df["ExpCorr"] = exp_corrs 
output_df["ExpPvalue"] = exp_pvalues

output_df["Statistic"] = stat
output_df["Pvalue"] = pvalue
output_df["FDR"] = adjusted_pvalue
output_df = output_df.sort_values(["FDR", "Pvalue"])

output_df.to_csv(
    OUTPUT_DIR_PATH.rstrip("/") + \
    "/{}_ztest.csv".format(CORRELATION),
    sep=",",
    index=None
)
