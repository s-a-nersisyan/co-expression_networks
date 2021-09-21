import numpy as np
import pandas as pd
import scipy.stats

import itertools

import sys
import time
import tqdm
import json

import seaborn as sns
import matplotlib.pyplot as plt

# Import python package
import core.extern
import core.utils

CORR_LEFT_BOUND = -1. + 1e-6
CORR_RIGHT_BOUND = 1. - 1e-6
MAX_DISTANCE = 1e10

# Arg parser
import argparse

# Cluster
from sklearn.cluster import \
    AgglomerativeClustering
from sklearn import manifold
from scipy.cluster import \
    hierarchy
from scipy.spatial import \
    distance

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("config_path")
parser.add_argument("-r", "--reduced", action="store_true")
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

REDUCED = args.reduced


# Main part
data_df = pd.read_csv(DATA_PATH, sep=",", index_col=0)
description_df = pd.read_csv(DESCRIPTION_PATH, sep=",")

if REDUCED:
    analysis_df = pd.read_csv(
        OUTPUT_DIR_PATH.rstrip("/") + \
        "/{}_binom.csv".format(CORRELATION),
        sep=","
    )

if (CORRELATION == "spearman"):
    correlation = core.extern.spearmanr
else:
    correlation = core.extern.pearsonr

interaction_df = None
source_indexes = None
target_indexes = None

# These corrs can be stored from
# the first step of the analysis
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
)

print("Graph calculations")
if REDUCED:
    significant_molecules = analysis_df["Molecule"]
    significant_indexes = core.utils.get_num_ind(
        data_df.index.to_list(),
        significant_molecules
    )
    significant_indexes = np.array(significant_indexes)
else:
    significant_indexes = np.arange(len(data_df))

ref_corrs = core.extern.quadrate(
    ref_corrs,
    significant_indexes,
    len(data_df)
) 

exp_corrs = core.extern.quadrate(
    exp_corrs,
    significant_indexes,
    len(data_df)
)

stat, pvalue = core.extern.ztest(
    ref_corrs.astype("float32"), np.zeros(len(ref_corrs), dtype="int32") +
        len(description_df.loc[description_df["Group"] == REFERENCE_GROUP]),
    exp_corrs.astype("float32"), np.zeros(len(exp_corrs), dtype="int32") +
        len(description_df.loc[description_df["Group"] == EXPERIMENTAL_GROUP]),
    correlation=CORRELATION,
    alternative=ALTERNATIVE,
    process_num=PROCESS_NUMBER
)

cluster_data = core.extern.quadrate(
    stat,
    significant_indexes,
    len(data_df)
)

cluster_df = pd.DataFrame(
    cluster_data,
    columns=data_df.index.to_list()
)

if REDUCED:
    cluster_df["Molecule"] = significant_molecules
else:
    cluster_df["Molecule"] = data_df.index.to_list()

print("Cluster process")
cluster_df = cluster_df.set_index("Molecule")
plot = sns.clustermap(cluster_df)
plot.savefig(
    OUTPUT_DIR_PATH.rstrip("/") + \
    "/{}_cluster.png".format(CORRELATION)
)

print("Store process")
cluster_df.to_csv(
    OUTPUT_DIR_PATH.rstrip("/") + \
    "/{}_cluster.csv".format(CORRELATION),
    sep=",", index=None
)
