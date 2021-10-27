import pandas as pd
import numpy as np
import tqdm
import sys

import matplotlib.pyplot as plt
import seaborn as sns

import core.extern

from scipy.stats import pearsonr

SEED = 4327

# Normalized data
GENE_VAR = 1.
GENE_MEAN = 2.
BOOT_ITERATION = 100
REPEATS_NUMBER = 10

np.random.seed(SEED)


# Data generation
def generate_sample(
    sample_size,
    gene_number,
    target_correlation,
    diff_target_percent,
    negative_flag=None
):
    diff_target_num = int(
        gene_number * diff_target_percent
    )
    covariation_matrix = np.zeros(
        (gene_number, gene_number),
        dtype="float32"
    )
    covariation_matrix[
        0:diff_target_num,
        0:diff_target_num
    ] = target_correlation

    for i in range(gene_number):
        covariation_matrix[i, i] = GENE_VAR

    if negative_flag:
        covariation_matrix[1:, 0] *= -1
        covariation_matrix[0, 1:] *= -1
    
    means = np.zeros(gene_number) + GENE_MEAN

    sample = np.random.multivariate_normal(
        means, covariation_matrix, size=sample_size
    ).transpose()

    return sample

# Generate diff expression sample
def get_pvalue(
    sample_size,
    gene_number,
    target_correlation,
    diff_target_percent,
    correlation,
    score,
    alternative
):
    first_sample = generate_sample(
        sample_size,
        gene_number,
        np.abs(target_correlation),
        diff_target_percent,
        negative_flag=False
    )

    second_sample = generate_sample(
        sample_size,
        gene_number,
        np.abs(target_correlation),
        diff_target_percent,
        negative_flag=True
    )
    
    data = np.concatenate((first_sample, second_sample), axis=1)
    source_indexes = np.zeros(gene_number - 1, dtype="int32")
    target_indexes = np.arange(gene_number - 1, dtype="int32") + 1
    reference_indexes = np.arange(sample_size, dtype="int32")
    experimental_indexes = np.arange(sample_size, dtype="int32") + sample_size

    data_df = pd.DataFrame(
        data=data,
        index=np.arange(gene_number),
        columns=np.arange(2 * sample_size)
    )

    indexes, scores, pvalues = \
    core.extern.score_pipeline(
        data_df,
        reference_indexes,
        experimental_indexes,
        source_indexes,
        target_indexes,
        correlation=correlation,
        score=score,
        alternative=alternative,
        repeats_num=BOOT_ITERATION,
        process_num=1,
        numerical_index=True
    )

    return pvalues[0]


# Grid generation
# sample_sizes = [i for i in range(5, 10)]
# sample_sizes.extend([i for i in range(10, 50, 5)])
# sample_sizes.extend([i for i in range(50, 100, 10)])
# sample_sizes.extend([i for i in range(100, 500, 50)])
# sample_sizes.extend([i for i in range(500, 1100, 100)])
# sample_sizes.append(1280)
# sample_sizes = np.array(sample_sizes)
gene_number = 1000

target_correlations = [0.7, 0.5, 0.3]
target_correlations = np.array(target_correlations)
print(target_correlations)

# target_correlations = np.array(target_correlations)[::-1]
# target_correlations = np.array(target_correlations)

# diff_target_percents = np.array([0.5, 0.1, 0.01])
# diff_target_percents = [0.5]
# diff_target_percents = [0.1]
diff_target_percents = [0.01]

correlation = "spearman"
score = "mean"
alternative = "two-sided"

pvalue_df = pd.DataFrame(
    columns=["Pvalue", "Sample size", "Correlation", "Diff percent"]
)

for dtp in diff_target_percents:
    for t_corr in target_correlations:
        print("Target corr: {}; Diff target percent: {};".format(
            t_corr, dtp
        ))
        
        pvalues = []
        sizes = []
        for ss in tqdm.tqdm(sample_sizes):
            # print("Sample size: {}".format(ss))
            # for repeat in tqdm.tqdm(range(REPEATS_NUMBER)):
            for repeat in range(REPEATS_NUMBER):
                pv = get_pvalue(
                    ss,
                    gene_number,
                    t_corr,
                    dtp,
                    correlation,
                    score,
                    alternative
                )

                pvalues.append(pv)
                sizes.append(ss)
            
            df = pd.DataFrame(columns=["Pvalue", "Sample size", "Correlation", "Diff percent"])
            df["Pvalue"] = pvalues
            df["Sample size"] = sizes
            df["Correlation"] = t_corr
            df["Diff percent"] = dtp

            pvalue_df = pd.concat([pvalue_df, df])
            pvalue_df.to_csv("synthetic/add_{}_{}.csv".format(
                score, dtp
            ), sep=",", index=None)

            sns.lineplot(data=pvalue_df, x="Sample size", y="Pvalue", hue="Correlation")
            plt.xscale("log")
            plt.savefig("synthetic/add_{}_{:.2f}.png".format(
                score, dtp
            ))

            plt.close()
