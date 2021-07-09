import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tqdm

import sys
sys.path.append("../")

from core.correlation import *

first_dist = np.random.multivariate_normal(
    [0, 0], [[1, 0.7], [0.7, 1]], 100
)
first_target = first_dist[:, 0]
first_sample = first_dist[:, 1]

second_dist = np.random.multivariate_normal(
    [0, 0], [[1, -0.3], [-0.3, 1]], 10
)
second_target = second_dist[:, 0]
second_sample = second_dist[:, 1]

print("Analytic computations")
test, pvalue = pearsonr_diff_test(
    first_target, [first_sample],
    second_target, [second_sample],
)

print(pvalue)
