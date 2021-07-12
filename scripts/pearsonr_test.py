import numpy as np
import pandas as pd

import tqdm

import sys

from core.CorrTest import *
from config import *

from CorrUtils import pearsonr
from CorrUtils import UNDEFINED_CORR_VALUE

first_df = pd.read_csv(FIRST_DATA_PATH, sep=",", index_col=0)
second_df = pd.read_csv(SECOND_DATA_PATH, sep=",", index_col=0)

network_df = None
if (NETWORK_PATH):
    network_df = pd.read_csv(NETWORK_PATH, sep=",", index_col=0)

def get_numeric_indexes(indexes, str_indexes):
    index_numbers = np.arange(len(indexes))

    return np.array([
        index_numbers[
            np.where(indexes == s_ind)
        ][-1] for s_ind in str_indexes
    ])

def get_pearson_correlations(df, source_indexes, target_indexes):
    df_indexes = np.array(df.index)
    df_data = df.to_numpy().astype("float32")

    source_num_indexes = get_numeric_indexes(df_indexes, source_indexes).astype("int32")
    target_num_indexes = get_numeric_indexes(df_indexes, target_indexes).astype("int32")
    corrs = pearsonr(
        df_data,
        source_num_indexes,
        target_num_indexes,
        1
    )

    corrs[corrs == UNDEFINED_CORR_VALUE] = None
    return corrs

network_df = network_df.iloc[:2]
print(network_df)

print(get_pearson_correlations(
    first_df,
    network_df["Source"],
    network_df["Target"]
))
