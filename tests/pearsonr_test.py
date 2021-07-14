import numpy as np
import pandas as pd
import scipy.stats

import sys
import time
import tqdm

# Import python package
import core
from config import *

# Import cpp package
# sys.path.append("build/")
# import correlation_computations

TOP_SIZE = 10**6

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
    df_data = df.to_numpy().astype("float32")

    source_num_indexes = get_numeric_indexes(df_indexes, source_indexes).astype("int32")
    target_num_indexes = get_numeric_indexes(df_indexes, target_indexes).astype("int32")

    start = time.time()
    corrs = core.pearsonr(
        df_data,
        source_num_indexes,
        target_num_indexes,
        PROCESS_NUMBER
    )
    print("Corr computational time: ", time.time() - start)

    corrs[corrs == core.UNDEFINED_CORR_VALUE] = None
    return corrs

network_df = network_df.iloc[:TOP_SIZE]

start = time.time()
corrs = get_pearson_correlations(
    first_df,
    network_df["Source"],
    network_df["Target"]
)
print(corrs[:10])
print(corrs[-10:])

print("Whole time: ", time.time() - start)

for i in range(10):
    print(scipy.stats.pearsonr(
        first_df.loc[network_df.iloc[i]["Source"]],
        first_df.loc[network_df.iloc[i]["Target"]]
    )[0])
