import numpy as np
import pandas as pd
import scipy.stats

import sys
import time
import tqdm

# Import python package
import core
from config import *

# Arg parser
import argparse


# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("-o", "--oriented", action="store_true")
args = parser.parse_args()
ORIENTED = args.oriented
if not INTERACTION_PATH:
    ORIENTED = False

# The main part
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
report_occurrence = get_occurrence(list(report_df["Source"]))
if not ORIENTED:
    report_occurrence.update(
            get_occurrence(list(report_df["Target"]))
    )

report_interaction_number = 0
for index in report_occurrence:
    report_interaction_number += report_occurrence[index]

# Description indexing
if INTERACTION_PATH:
    description_df = pd.read_csv(DESCRIPTION_PATH, sep=",")
    
    initial_occurrence = get_occurrence(list(description_df["Source"]))
    if not ORIENTED:
        initial_occurrence.update(
                get_occurrence(list(description_df["Target"]))
        )
    
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
output_df = output_df.sort_values(["FDR", "Pvalue"])
output_df.to_csv(
        OUTPUT_DIR_PATH.rstrip("/") +
        "/{}_analisis.csv".format(CORRELATION),
        sep=",",
        index=None
)

output_df[output_df["FDR"] < FDR_THRESHOLD]["Molecule"].to_csv(
        OUTPUT_DIR_PATH.rstrip("/") +
        "/{}_molecules.txt".format(CORRELATION),
        sep=" ",
        index=None
)
