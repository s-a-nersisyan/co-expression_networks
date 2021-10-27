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

interaction_df = pd.read_csv(INTERACTION_PATH, sep=",")

data_molecules = set(data_df.index.to_list())
interaction_df = interaction_df[interaction_df["Source"].isin(data_molecules)]
interaction_df = interaction_df[interaction_df["Target"].isin(data_molecules)]

reference_indexes = description_df.loc[
    description_df["Group"] == REFERENCE_GROUP,
    "Sample"
].to_list()
experimental_indexes = description_df.loc[
    description_df["Group"] == EXPERIMENTAL_GROUP,
    "Sample"
].to_list()

# Test mode
interaction_df = interaction_df.iloc[:10]

source_indexes = interaction_df["Source"]
target_indexes = interaction_df["Target"]

data_molecules = data_molecules.intersection(
    set(source_indexes) | set(target_indexes)
)
data_df = data_df.loc[data_df.index.isin(data_molecules)]

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

# The hypothesis check
print("Test phase")
stat, pvalue = core.extern.ztest(
    ref_corrs.astype("float32"),
    len(reference_indexes),
    exp_corrs.astype("float32"),
    len(experimental_indexes),
    correlation=CORRELATION,
    alternative=ALTERNATIVE,
    process_num=PROCESS_NUMBER
)

adjusted_pvalue = pvalue * len(pvalue) / \
    scipy.stats.rankdata(pvalue)
adjusted_pvalue[adjusted_pvalue > 1] = 1
adjusted_pvalue = adjusted_pvalue.flatten()

# Generate report
print("Report phase")
output_df = pd.DataFrame(interaction_df)

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
