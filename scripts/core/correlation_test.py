import numpy as np

from scipy.stats import t
from scipy.stats import norm
from scipy.stats import spearmanr
from scipy.stats import pearsonr

from .utils import *
from .correlation_proba import *

def pearsonr_test(
    first_corrs, first_sizes,
    second_corrs, second_sizes,
    alternative="two-sided",
    values="tp"
):
    first_corrs = np.array(first_corrs)
    first_sizes = np.array(first_sizes)

    second_corrs = np.array(second_corrs)
    second_sizes = np.array(second_sizes)

    first_ss = pearsonr_std(first_corrs, first_sizes)
    second_ss = pearsonr_std(second_corrs, second_sizes)

    test = np.arctanh(first_corrs) - np.arctanh(second_corrs)
    pvalue = None

    if (alternative == "less"):
        pvalue = norm.cdf(test,
            scale=np.sqrt(first_ss**2 + second_ss**2))
    elif (alternative == "greater"):
        pvalue = 1 - norm.cdf(test,
            scale=np.sqrt(first_ss**2 + second_ss**2))
    elif (alternative == "two-sided"):
        pvalue = 2 * norm.cdf(-np.abs(test),
            scale=np.sqrt(first_ss**2 + second_ss**2))

    result = []
    if ("t" in values):
        result.append(test)
    if ("p" in values):
        result.append(pvalue)

    return result
