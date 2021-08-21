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
if not INTERACTION_PATH:
    ORIENTED = False

# Main part

# TODO: if interaction err was rased
# the following string is superfluous 
data_df = pd.read_csv(DATA_PATH, sep=",", index_col=0)

report_df = pd.read_csv(
        OUTPUT_DIR_PATH.rstrip("/") + 
        "/{}_report.csv".format(CORRELATION),
        sep=","
)

def get_occurrence(array):
    occurrence = {}
    for elem in array:
        if elem in occurrence:
            occurrence[elem] += 1
        else:
            occurrence[elem] = 1
    
    return occurrence

# Report indexing
if ORIENTED:
    report_occurrence = get_occurrence(list(report_df["Source"]))
else:
    ll = list(report_df["Source"])
    ll.extend(list(report_df["Target"]))
    report_occurrence = get_occurrence(ll)

report_interaction_number = 0
for index in report_occurrence:
    report_interaction_number += report_occurrence[index]

# Description indexing
if INTERACTION_PATH:
    interaction_df = pd.read_csv(INTERACTION_PATH, sep=",")

    # TODO: raise error if a "Source", "Target" interaction pair
    # is not found in data molecules
    data_molecules = set(data_df.index.to_list())
    interaction_df = interaction_df[interaction_df["Source"].isin(data_molecules)]
    interaction_df = interaction_df[interaction_df["Target"].isin(data_molecules)]
    
    # TODO: copies should be removed
    if ORIENTED:
        initial_occurrence = get_occurrence(list(interaction_df["Source"]))
    else:
        ll = list(interaction_df["Source"])
        ll.extend(list(interaction_df["Target"]))
        initial_occurrence = get_occurrence(ll)
    
    initial_interaction_number = 0
    for index in initial_occurrence:
        initial_interaction_number += initial_occurrence[index]

else:
    data_df = pd.read_csv(DATA_PATH, sep=",")
    
    initial_occurrence = dict(report_occurrence)
    for index in initial_occurrence:
        initial_occurrence[index] = len(data_df)
     
    initial_interaction_number = len(data_df) * (len(data_df) - 1) // 2

# Analisis
output_df = pd.DataFrame()
output_df["Molecule"] = [index for index in report_occurrence]
output_df["Diff"] = [report_occurrence[molecule] for molecule in output_df["Molecule"]]
output_df["Total"] = [initial_occurrence[molecule] for molecule in output_df["Molecule"]]
output_df["Proportion"] = output_df["Diff"] / output_df["Total"]
output_df["Pvalue"] = 1 - scipy.stats.binom.cdf(
    output_df["Diff"],
    output_df["Total"],
    report_interaction_number / initial_interaction_number
)

adjusted_pvalue = np.array(output_df["Pvalue"] * len(output_df["Pvalue"]) / \
    scipy.stats.rankdata(output_df["Pvalue"]))
adjusted_pvalue[adjusted_pvalue > 1] = 1
adjusted_pvalue = adjusted_pvalue.flatten()
output_df["FDR"] = adjusted_pvalue
output_df = output_df[output_df["FDR"] < FDR_THRESHOLD]

output_df = output_df.sort_values(["FDR", "Pvalue"])
output_df.to_csv(
        OUTPUT_DIR_PATH.rstrip("/") +
        "/{}_analysis.csv".format(CORRELATION),
        sep=",",
        index=None
)

# output_df[output_df["FDR"] < FDR_THRESHOLD]["Molecule"].to_csv(
#         OUTPUT_DIR_PATH.rstrip("/") +
#         "/{}_molecules.txt".format(CORRELATION),
#         sep=" ",
#         index=None
# )
