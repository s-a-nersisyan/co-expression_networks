import numpy as np
import pandas as pd
import scipy.stats

import itertools

import sys
import time
import tqdm

import seaborn as sns

# Import python package
import core
from config import *

CORR_LEFT_BOUND = -1 + 1e-6
CORR_RIGHT_BOUND = 1 - 1e-6


data_df = pd.read_csv(DATA_PATH, sep=",", index_col=0)
description_df = pd.read_csv(DESCRIPTION_PATH, sep=",")
analisis_df = pd.read_csv(
    OUTPUT_DIR_PATH.rstrip("/") + \
    "/{}_analysis.csv".format(CORRELATION),
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
if (INTERACTION_PATH):
    interaction_df = pd.read_csv(INTERACTION_PATH, sep=",", index_col=0)
    source_indexes = interaction_df["Source"]
    target_indexes = interaction_df["Target"]

else:
    source_indexes=None
    target_indexes=None

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
significant_molecules = analisis_df["Molecule"]
significant_indexes = core.get_num_ind(
    data_df.index.to_list(),
    significant_molecules
)

significant_indexes = np.array(significant_indexes)

if not INTERACTION_PATH:
    sign_ref_corrs = core.paired_reshape(
        ref_corrs,
        significant_indexes,
        len(data_df)
    ) 
    
    sign_exp_corrs = core.paired_reshape(
        exp_corrs,
        significant_indexes,
        len(data_df)
    )
    
    sign_ref_corrs = core.bound(
        sign_ref_corrs,
        CORR_LEFT_BOUND, CORR_RIGHT_BOUND
    )
    
    sign_exp_corrs = core.bound(
        sign_exp_corrs,
        CORR_LEFT_BOUND, CORR_RIGHT_BOUND
    )

    print("Save cluster table")
    cluster_data = np.arctanh(sign_ref_corrs) - \
        np.arctanh(sign_exp_corrs) 
    np.save(
        OUTPUT_DIR_PATH.rstrip("/") + \
        "/{}_cluster.npy".format(CORRELATION),
        cluster_data
    )

    cluster_df = pd.DataFrame(
        cluster_data,
        columns=data_df.index.to_list()
    )
    cluster_df["Molecule"] = significant_molecules
    cluster_df.to_csv(
        OUTPUT_DIR_PATH.rstrip("/") + \
        "/{}_cluster.csv".format(CORRELATION),
        sep=",", index=None
    )
    print("End of save process")

    cluster_df = cluster_df.set_index("Molecule")
    plot = sns.clustermap(cluster_df)
    plot.savefig(
        OUTPUT_DIR_PATH.rstrip("/") + \
        "/{}_cluster.png".format(CORRELATION)
    )

else :
    interaction_graph = dict()
    corr_numeration =  dict()
    sm_set = set(significant_molecules)
    for ind, row in description_df.iterrows():
        if not (row["Source"] in sm_set):
            continue
        
        if not (row["Source"] in interaction_graph):
            interaction_graph[row["Source"]] = list()
        interaction_graph[row["Source"]].append(row["Target"])
        
        if not ((row["Source"], row["Target"]) in corr_numeration):
            corr_numeration[(row["Source"], row["Target"])] = ind
    del sm_set

    distances = {}
    for  first, second in itertools.combinations(significant_indexes):
        intersection = set(interaction_graph[first]).interaction(
            set(interaction_graph[second])
        )
        
        distances[(first, second)] = 0.
        for third in intersection:
            first_corr_ind = corr_numeration[(first, third)]
            second_corr_ind = corr_numeration[(second, third)]
            
            first_stat = np.arctanh(ref_corrs[first_corr_ind]) - \
                    np.arctanh(exp_corrs[first_corr_ind])
            second_stat = np.arctanh(ref_corrs[second_corr_ind]) - \
                    np.arctanh(exp_corrs[second_corr_ind])
            
            distances[(first, second)] += (first_stat - second_stat)**2
        
        distances[(first, second)] /= len(intersection)
        distances[(first, second)] = np.sqrt(distances[(first, second)])
    
    sources = []
    tagets = []
    dists = []
    for key in distances:
        source, target = *key
        sources.appen(source)
        target.append(target)
        dists.append(distances[(source, target)])
    
    cluster_df = pd.DataFrame()
    cluster_df["Source"] = sources
    cluster_df["Target"] = targets
    cluster_df["Distance"] = dists
    cluster_df.to_csv(
        OUTPUT_DIR_PATH.rstrip("/") + \
        "/{}_cluster.csv".format(CORRELATION),
        sep=",",
        index=None
    )
