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
parser.add_argument("-o", "--oriented", action="store_true")
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

ORIENTED = args.oriented

BOOTSTRAP_REPEATS = 10**3

def bootstrap(
    arr, statistic, size, 
    bootstrap_repeats=BOOTSTRAP_REPEATS
):
    result = []
    for i in range(BOOTSTRAP_REPEATS):
        sample = np.random.choice(
            arr, size,
            replace=True
        )

        result.append(statistic(sample))
    
    return np.array(result)


# Main part
report_df = pd.read_csv(
        OUTPUT_DIR_PATH.rstrip("/") + 
        "/{}_report.csv".format(CORRELATION),
        sep=","
)
report_df = report_df[report_df["Pvalue"] < FDR_THRESHOLD]
stat = np.abs(np.array(report_df["Statistic"]).flatten())

pvalues = dict()
means = dict()
sizes = dict()

if ORIENTED:
    indexes = ["Source"]
else:
    indexes = ["Source", "Target"]

for ind in indexes:
    for molecule in tqdm.tqdm(set(report_df[ind])):
        molecule_stats = np.abs(np.array(report_df.loc[
            report_df[ind] == molecule, "Statistic"
        ]).flatten())

        mean_dist = bootstrap(
            stat, lambda x: sum(x) / len(x),
            len(molecule_stats),
            bootstrap_repeats=BOOTSTRAP_REPEATS
        )

        means[molecule] = np.mean(molecule_stats)
        sizes[molecule] = len(molecule_stats)
        pvalues[molecule] = np.sum(mean_dist > means[molecule]) / len(mean_dist)

output_df = pd.DataFrame()
output_df["Molecule"] = [molecule for molecule in means]
output_df["Mean"] = [means[molecule] for molecule in means]
output_df["Total"] = [sizes[molecule] for molecule in means]
output_df["Pvalue"] = [pvalues[molecule] for molecule in pvalues] 

adjusted_pvalue = np.array(output_df["Pvalue"] * len(output_df["Pvalue"]) / \
    scipy.stats.rankdata(output_df["Pvalue"]))
adjusted_pvalue[adjusted_pvalue > 1] = 1
adjusted_pvalue = adjusted_pvalue.flatten()
output_df["FDR"] = adjusted_pvalue

output_df = output_df.sort_values(["FDR", "Pvalue"])
output_df.to_csv(
        OUTPUT_DIR_PATH.rstrip("/") +
        "/{}_aggregation.csv".format(CORRELATION),
        sep=",",
        index=None
)
