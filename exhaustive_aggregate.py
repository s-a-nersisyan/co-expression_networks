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
EXPERIMENTAL_GROUP = config["experimnetal_group"]

CORRELATION = config["correlation"]
ALTERNATIVE = config["alternative"]
PROCESS_NUMBER = config["process_number"]

FDR_THRESHOLD = config["fdr_treshold"]

# in exhaustive mode we don't use
# graph structure
INTERACTION_PATH = None

# Bootstrap constant and method
BOOTSTRAP_REPEATS = 10**5
CHUNK_SIZE = 10**3

def statistic(arr):
    return np.median(arr, axis=1).flatten()

def bootstrap(
    arr, statistic, size, 
    bootstrap_repeats=BOOTSTRAP_REPEATS,
    chunk_size=CHUNK_SIZE
):
    result = []
    repeats_num = (bootstrap_repeats - 1) // chunk_size + 1
    for i in range(repeats_num):
        if i < repeats_num - 1:
            indexes = np.randint(
                len(arr), size=(len(arr), chunk_size)
            )
        elif bootstrap_repeats % chunk_size > 0:
            indexes = np.randint(
                len(arr),
                size=(len(arr), bootstrap_repeats % chunk_size)
            )

        sample = arr[indexes]
        
        if (chunk_size > 1):
            result.extend(statistic(sample))
        else:
            result.append(statistic(sample))
            

    for i in tqdm.tqdm(range(BOOTSTRAP_REPEATS)):
        sample = np.random.choice(
            arr, size,
            replace=True
        )

        result.append(statistic(sample))
    
    return np.array(result)


# Main part
stat = np.abs(np.load(OUTPUT_DIR_PATH.rstrip("/") + \
    "/{}_report_stat.npy".format(CORRELATION)
))
molecule_num = int((1 + np.sqrt(1 + 8 * len(stat))) / 2)

# Calculate mean values of a molecule
print("Mean phase")
mean = np.zeros(molecule_num)
mean = np.mean(core.extern.quadrate(
    stat, np.arange(molecule_num),
    molecule_num
), axis = 1)

# Distribution to calculate pvalue
print("Bootstrap phase")
mean_distribution = bootstrap(
    stat,
    lambda x: np.sum(x) / len(x),
    molecule_num,
    bootstrap_repeats=BOOTSTRAP_REPEATS
)

# Calculate pvalue
print("Pvalue")
pvalue = np.zeros(molecule_num)
for ind, mn in tqdm.tqdm(enumerate(mean)): 
    pvalue[ind] = np.sum(mean_distribution > mn) / len(mean_distribution)

adjusted_pvalue = pvalue * len(pvalue) / \
    scipy.stats.rankdata(pvalue)
adjusted_pvalue[adjusted_pvalue > 1] = 1
adjusted_pvalue = adjusted_pvalue.flatten()

# Save the result
molecule = pd.read_csv(DATA_PATH, sep=",", index_col=0).index.to_numpy()

output_df = pd.DataFrame()
output_df["Molecule"] = molecule
output_df["Mean"] = mean
output_df["Total"] = [molecule_num] * len(molecule)
output_df["Pvalue"] = pvalue
output_df["FDR"] = adjusted_pvalue

output_df = output_df.sort_values(["FDR", "Pvalue"])
output_df.to_csv(
    OUTPUT_DIR_PATH.rstrip("/") +
    "/{}_aggregation.csv".format(CORRELATION),
    sep=",",
    index=None
)
