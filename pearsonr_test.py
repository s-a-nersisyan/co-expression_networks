import numpy as np
import pandas as pd
import scipy.stats

import sys
import time
import tqdm

# Import python package
import core
from config import *


TOP_SIZE = 10**3


first_df = pd.read_csv(FIRST_DATA_PATH, sep=",", index_col=0)
second_df = pd.read_csv(SECOND_DATA_PATH, sep=",", index_col=0)

network_df = None
if (NETWORK_PATH):
    network_df = pd.read_csv(NETWORK_PATH, sep=",", index_col=0)

def get_numeric_indexes(indexes, str_indexes):
    indexes = {ind: num for num, ind in tqdm.tqdm(enumerate(indexes))}
    return np.array([
         indexes[s_ind] for s_ind in tqdm.tqdm(str_indexes)
    ])

def get_pearson_correlations(df, source_indexes, target_indexes):
    df_indexes = np.array(df.index)
    df_data = df.to_numpy(copy=True).astype("float32")

    source_num_indexes = get_numeric_indexes(
        df_indexes, source_indexes
    ).astype("int32")
    target_num_indexes = get_numeric_indexes(
        df_indexes, target_indexes
    ).astype("int32")

    corrs = core.pearsonr(
        df_data,
        source_num_indexes,
        target_num_indexes,
        PROCESS_NUMBER
    )

    corrs[corrs == core.UNDEFINED_CORR_VALUE] = None
    return corrs

network_df = network_df.iloc[:TOP_SIZE]

first_corrs = get_pearson_correlations(
    first_df,
    network_df["Source"],
    network_df["Target"]
)

second_corrs = get_pearson_correlations(
    second_df,
    network_df["Source"],
    network_df["Target"]
)

stat, pvalue = core.pearsonr_test(
    first_corrs, len(first_df.iloc[0]),
    second_corrs, len(second_df.iloc[0])
)

adjusted_pvalue = pvalue * len(pvalue) / \
    scipy.stats.rankdata(pvalue)

# Generate report
output_df = pd.DataFrame(network_df)
output_df["Statistic"] = stat
output_df["Pvalue"] = pvalue
output_df["Adjusted"] = adjusted_pvalue
output_df.to_csv(OUTPUT_DIR_PATH.rstrip("/") + "/report.csv", sep=",")
