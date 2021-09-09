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
import core

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

MAX_DISTANCE = 1e10

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


# Main part
data_df = pd.read_csv(DATA_PATH, sep=",", index_col=0)
description_df = pd.read_csv(DESCRIPTION_PATH, sep=",")
analysis_df = pd.read_csv(
    OUTPUT_DIR_PATH.rstrip("/") + \
    "/{}_binom.csv".format(CORRELATION),
    sep=","
)

# Test mode
# data_df = data_df.iloc[:100]
# End of Test mode

if (CORRELATION == "spearman"):
    correlation = core.spearmanr
else:
    correlation = core.pearsonr

interaction_df = None
interaction_df = pd.read_csv(INTERACTION_PATH, sep=",")

# TODO: raise error if a "Source",
# "Target" interaction pair
# is not found in data molecules
data_molecules = set(data_df.index.to_list())
interaction_df = interaction_df[interaction_df["Source"].isin(data_molecules)]
interaction_df = interaction_df[interaction_df["Target"].isin(data_molecules)]

source_indexes = interaction_df["Source"]
target_indexes = interaction_df["Target"]

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
significant_molecules = analysis_df["Molecule"]
significant_indexes = core.get_num_ind(
    data_df.index.to_list(),
    significant_molecules
)
significant_indexes = np.array(significant_indexes)

stat, pvalue = core.corr_diff_test(
    ref_corrs.astype("float32"), np.zeros(len(ref_corrs), dtype="int32") +
        len(description_df.loc[description_df["Group"] == REFERENCE_GROUP]),
    exp_corrs.astype("float32"), np.zeros(len(exp_corrs), dtype="int32") +
        len(description_df.loc[description_df["Group"] == EXPERIMENTAL_GROUP]),
    correlation=CORRELATION,
    alternative=ALTERNATIVE,
    process_num=PROCESS_NUMBER
)
stat = np.abs(stat)

interaction_graph = dict()
corr_numeration =  dict()
sm_set = set(significant_molecules)

for ind, (_, row) in enumerate(interaction_df.iterrows()):
    if not ORIENTED and (row["Target"] in sm_set):
        if not (row["Target"] in interaction_graph):
            interaction_graph[row["Target"]] = list()
        interaction_graph[row["Target"]].append(row["Source"])
        
        if not ((row["Target"], row["Source"]) in corr_numeration):
            corr_numeration[(row["Target"], row["Source"])] = ind

    if not (row["Source"] in sm_set):
        continue
    
    if not (row["Source"] in interaction_graph):
        interaction_graph[row["Source"]] = list()
    interaction_graph[row["Source"]].append(row["Target"])
    
    if not ((row["Source"], row["Target"]) in corr_numeration):
        corr_numeration[(row["Source"], row["Target"])] = ind

del sm_set

distances = {}
for first, second in itertools.combinations(significant_molecules, 2):
    intersection = set(interaction_graph[first]).intersection(
        set(interaction_graph[second])
    )
    
    distances[(first, second)] = 0.
    for third in intersection:
        first_corr_ind = corr_numeration[(first, third)]
        second_corr_ind = corr_numeration[(second, third)]
        
        # first_stat = np.arctanh(ref_corrs[first_corr_ind]) - \
        #         np.arctanh(exp_corrs[first_corr_ind])
        # second_stat = np.arctanh(ref_corrs[second_corr_ind]) - \
        #         np.arctanh(exp_corrs[second_corr_ind])
        
        first_stat = stat[first_corr_ind]
        second_stat = stat[second_corr_ind]

        distances[(first, second)] += (first_stat - second_stat)**2
   
    distances[(first, second)] = np.sqrt(distances[(first, second)])
    
    if len(intersection) > 0: 
        distances[(first, second)] /= len(intersection)
    else:
        distances[(first, second)] = MAX_DISTANCE

sources = []
targets = []
dists = []

print("Store process")
for key in distances:
    source, target = key
    sources.append(source)
    targets.append(target)
    dists.append(distances[(source, target)])

cluster_df = pd.DataFrame()
cluster_df["Source"] = sources
cluster_df["Target"] = targets
cluster_df["Distance"] = dists
cluster_df = cluster_df.sort_values(by="Distance")
cluster_df.to_csv(
    OUTPUT_DIR_PATH.rstrip("/") + \
    "/{}_cluster.csv".format(CORRELATION),
    sep=",",
    index=None
)

print("Cluster process")
distance_matrix = cluster_df["Distance"].to_numpy()

# TODO: this procedure may be superfluous 
np.nan_to_num(distance_matrix, nan=MAX_DISTANCE, copy=False)

dendrogram = hierarchy.dendrogram(hierarchy.linkage(
    distance_matrix, method="ward"    
))

plt.savefig(
    OUTPUT_DIR_PATH.rstrip("/") + \
    "/{}_cluster.png".format(CORRELATION)
)
plt.close()

print("Cluster plot")
distance_matrix = distance.squareform(
    distance_matrix,
    force="tomatrix"
)

model = manifold.MDS(
    n_components=2,
    dissimilarity="precomputed"
)
model.fit(distance_matrix)
coords = model.fit_transform(distance_matrix)
plt.scatter(coords[:, 0], coords[:, 1])

plt.savefig(
    OUTPUT_DIR_PATH.rstrip("/") + \
    "/{}_mds.png".format(CORRELATION)
)
