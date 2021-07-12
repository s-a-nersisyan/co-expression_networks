import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import pearsonr

import tqdm

import sys
sys.path.append("../")

from core.correlation import *
from core.utils import *

np.random.seed(11)

first_dist = np.random.multivariate_normal(
    [0, 0], [[1, 0.2], [0.2, 1]], 10**2
)
first_target = first_dist[:, 0]
first_sample = first_dist[:, 1]

second_dist = np.random.multivariate_normal(
    [0, 0], [[1, 0.45], [0.45, 1]], 10**2
)
second_target = second_dist[:, 0]
second_sample = second_dist[:, 1]

print("Analytic computations")
test, pvalue = pearsonr_diff_test(
    first_target, [first_sample],
    second_target, [second_sample],
)
print(pvalue)


print("Bootstrap computations")
first_rs = pearsonr(first_target, first_sample)[0]
second_rs = pearsonr(second_target, second_sample)[0]
delta = np.arctanh(first_rs) - np.arctanh(second_rs)

first_dist = np.random.multivariate_normal(
    [0, 0], [[1, 0.8], [0.8, 1]], 10**2
)
first_target = first_dist[:, 0]
first_sample = first_dist[:, 1]

second_dist = np.random.multivariate_normal(
    [0, 0], [[1, 0.8], [0.8, 1]], 10**2
)
second_target = second_dist[:, 0]
second_sample = second_dist[:, 1]


first_rs_sample = np.arctanh([rs for rs, _ in bootstrap_sample(
    first_target,
    first_sample,
    statistic=pearsonr
)])

second_rs_sample = np.arctanh([rs for rs, _ in bootstrap_sample(
    second_target,
    second_sample,
    statistic=pearsonr
)])

pvalue = np.sum(
    np.abs(first_rs_sample - second_rs_sample) >\
    np.abs(delta)
)
pvalue /= len(first_rs_sample)
print(pvalue)
