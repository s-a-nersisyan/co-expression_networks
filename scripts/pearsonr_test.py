import numpy as np
import pandas as pd

import tqdm

import sys

from core.correlation import *
from config import *


first_df = pd.read_csv(FIRST_SAMPLE_PATH, sep=",", index_col=0)
second_df = pd.read_csv(SECOND_SAMPLE_PATH, sep=",", index_col=0)

network_df = None
if (NETWORK_PATH):
    network_df = pd.read_csv(NETWORK_PATH, sep=",", index_col=0)

result = []
def _run_process(target, sample):
    first_target = first_df.loc[target]
    first_sample = first_df.loc[sample]

    second_target = second_df.loc[target]
    second_sample = second_df.loc[sample]

    first_rs, second_rs, pvalue = pearsonr_diff_test(
        first_target, first_sample,
        second_target, second_sample,
        values="fsp"
    )

    for elem, f, s, p in zip(sample, first_rs, second_rs, pvalue):
        result.append([target, elem, f, s, p])

def run_process(indexes):
    df = network_df.iloc[indexes]
    for target in df["Target"].unique():
        _run_process(target, df[df["Target"] == target]["Sample"].unique())


network_df = network_df.sort_values(by=["Target", "Sample"]).iloc[:20]
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
