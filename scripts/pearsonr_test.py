import numpy as np
import pandas as pd

import tqdm

import sys

from core.correlation import *
from config import *
from pcorr import pearson_correlation


first_df = pd.read_csv(FIRST_target_PATH, sep=",", index_col=0)
second_df = pd.read_csv(SECOND_target_PATH, sep=",", index_col=0)

network_df = None
if (NETWORK_PATH):
    network_df = pd.read_csv(NETWORK_PATH, sep=",", index_col=0)

result = []
def _run_process(source, target):
    first_source = first_df.loc[source]
    first_target = first_df.loc[target]

    second_source = second_df.loc[source]
    second_target = second_df.loc[target]

    first_rs, second_rs, pvalue = pearsonr_diff_test(
        first_source, first_target,
        second_source, second_target,
        values="fsp"
    )

    for elem, f, s, p in zip(target, first_rs, second_rs, pvalue):
        result.append([source, elem, f, s, p])

def run_process(indexes):
    df = network_df.iloc[indexes]
    for source in df["source"].unique():
        _run_process(source, df[df["source"] == source]["target"].unique())


network_df = network_df.sort_values(by=["source", "target"]).iloc[:20]
batch_size = len(network_df) / PROCESS_NUMBER
for process_ind in range(PROCESS_NUMBER - 1):
    indexes = np.arange(
        batch_size * process_ind,
        batch_size * (process_ind + 1)
    )
    run_process(indexes)

indexes = np.arange(
    batch_size * (PROCESS_NUMBER - 1),
    len(network_df)
)
run_process(indexes)

print(result)
