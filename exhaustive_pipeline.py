import numpy as np
import pandas as pd
import scipy.stats

import sys
import time
import tqdm
import json

import time

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
EXPERIMENTAL_GROUP = config["experimnetal_group"]

CORRELATION = config["correlation"]
ALTERNATIVE = config["alternative"]
REPEATS_NUMBER = config["repeats_number"]
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

print("Pipeline")
start = time.time()
ref_corrs, exp_corrs, stat, pvalue, boot_pvalue = \
core.extern.ztest_pipeline(
    data_df,
    reference_indexes,
    experimental_indexes,
    source_indexes,
    target_indexes,
    correlation=CORRELATION,
    alternative=ALTERNATIVE,
    repeats_num=REPEATS_NUMBER,
    process_num=PROCESS_NUMBER
)
print(time.time() - start)

adjusted_pvalue = pvalue * len(pvalue) / \
    scipy.stats.rankdata(pvalue)
adjusted_pvalue[adjusted_pvalue > 1] = 1
adjusted_pvalue = adjusted_pvalue.flatten()

indexes = np.arange(len(adjusted_pvalue))

ref_corrs = ref_corrs[indexes]
exp_corrs = exp_corrs[indexes]
stat = stat[indexes]
pvalue = pvalue[indexes]
boot_pvalue = boot_pvalue[indexes]
adjusted_pvalue = adjusted_pvalue[indexes]
df_indexes = data_df.index.to_numpy()

np.save(OUTPUT_DIR_PATH.rstrip("/") + \
    "/{}_pipeline_stat.npy".format(CORRELATION),
    stat
)
np.save(OUTPUT_DIR_PATH.rstrip("/") + \
    "/{}_pipeline_pvalue.npy".format(CORRELATION),
    pvalue
)
np.save(OUTPUT_DIR_PATH.rstrip("/") + \
    "/{}_pipeline_bootpv.npy".format(CORRELATION),
    boot_pvalue
)
np.save(OUTPUT_DIR_PATH.rstrip("/") + \
    "/{}_pipeline_fdr.npy".format(CORRELATION),
    adjusted_pvalue
)

indexes = np.where(adjusted_pvalue < FDR_THRESHOLD)[0]

ref_corrs = ref_corrs[indexes]
exp_corrs = exp_corrs[indexes]
stat = stat[indexes]
pvalue = pvalue[indexes]
boot_pvalue = boot_pvalue[indexes]
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
output_df["Reference"] = ref_corrs 
output_df["Experimental"] = exp_corrs 
output_df["Statistic"] = stat
output_df["Pvalue"] = pvalue
output_df["Bootpv"] = boot_pvalue
output_df["FDR"] = adjusted_pvalue
output_df = output_df.sort_values(["FDR", "Pvalue"])

output_df.to_csv(
    OUTPUT_DIR_PATH.rstrip("/") + \
    "/{}_pipeline.csv".format(CORRELATION),
    sep=",",
    index=None
)
