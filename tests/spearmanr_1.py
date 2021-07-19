import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tqdm

# import "core" module
import sys
sys.path.append("../")
import core


first_target = np.random.normal(1.2, 1, 1000)
first_sample = np.random.standard_t(2, size=1000) + first_target

second_target = np.random.normal(1.2, 1, 1000)
second_sample = second_target**2 + 1 * np.random.normal(0, 1, 1000)

print("Analytic computations")
deltas = np.linspace(-2, 2, 1000)
probas = []
for d in tqdm.tqdm(deltas):
    probas.append(core.correlation_diff_cdf(
        first_target, [first_sample],
        second_target, [second_sample],
        d,
        correlation="spearman",
        alternative="less",
        method="analytic"
    ))
probas = np.array(probas)
plt.plot(deltas, probas, label="analytic")

print("Bootstrap computations")
deltas = np.linspace(-2, 2, 100)
probas = []
for d in tqdm.tqdm(deltas):
    probas.append(core.correlation_diff_cdf(
        first_target, [first_sample],
        second_target, [second_sample],
        d,
        correlation="spearman",
        alternative="less",
        method="bootstrap"
    ))
probas = np.array(probas)
plt.plot(deltas, probas, label="bootstrap")

plt.legend()
plt.savefig("./plots/{}.png".format(sys.argv[0].split(".")[0]))
