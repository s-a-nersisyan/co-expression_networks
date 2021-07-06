import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tqdm

import sys

from core.spearmanr import *


first_target = np.random.normal(1.2, 1, 1000)
first_sample = np.random.standard_t(2, size=1000) + first_target

second_target = np.random.normal(1.2, 1, 1000)
second_sample = second_target**2 + 1 * np.random.normal(0, 1, 1000)


print("Analytic computations")
deltas = np.linspace(-2, 2, 1000)
probas = []
for d in tqdm.tqdm(deltas):
    probas.append(spearmanr_diff_proba(
        first_target, [first_sample],
        second_target, [second_sample],
        d,
        alternative="less",
        method="analytic"
    ))
probas = np.array(probas)
plt.plot(deltas, probas, label="analytic")

print("Bootstrap computations")
deltas = np.linspace(-2, 2, 100)
probas = []
for d in tqdm.tqdm(deltas):
    probas.append(spearmanr_diff_proba(
        first_target, [first_sample],
        second_target, [second_sample],
        d,
        alternative="less",
        method="bootstrap"
    ))
probas = np.array(probas)
plt.plot(deltas, probas, label="bootstrap")

plt.legend()
plt.savefig("./tests/{}.png".format(sys.argv[0].split(".")[0]))
