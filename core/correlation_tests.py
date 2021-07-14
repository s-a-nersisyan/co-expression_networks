"""Tests of equal correlation hypothesis

The module contains methods testing hypothesis of
correlation equality
"""


import numpy as np
import scipy.stats
from .correlation_utils import \
    pearsonr_mean, pearsonr_std

def pearsonr_test(
    first_rs, first_size,
    second_rs, second_size,
    alternative="two-sided"
):
    """Check the hypothesis that pearson correlations
    are equal

    Parameters
    ----------
    first_rs: numerical value or list
        A sequence (potentially with one element)
        of sperman correlations.
    first_size: numerical value or list
        Sizes of samples that were used to compute
        "first_rs" correlation(s).
    second_rs: numerical value or list
        A sequence (potentially with one element)
        of sperman correlations.
    second_size: numerical value or list
        Sizes of samples that were used to compute
        "second_rs" correlation(s).
    alternative: "two-sided" (default), "less", "greater"
        Computes the probability of the following events:
        "two-sided" |arctanh(x1) - arctanh(x2)| >
        |arctanh(first_rs) - arctanh(second_rs)|,
        "less" arctanh(x1) - arctanh(x2) <=
        arctanh(first_rs) - arctanh(second_rs) ,
        "greater" arctanh(x1) - arctanh(x2) >
        arctanh(first_rs) - arctanh(second_rs).
    value: substring of "tp"
        If value contains "t" statistic
        np.arctanh(first_rs) - np.arctanh(second_rs)
        is included in the output.
        If value contains "p" pvalue of the test
        is included in the output.

    Returns
    -------
    numerical value or numpy.array respectively to the input
        Contains statistic and pvalue.
    """

    first_rs = np.array(first_rs)
    first_size = np.array(first_size)

    second_rs = np.array(second_rs)
    second_size = np.array(second_size)

    first_ss = pearsonr_std(first_rs, first_size)
    second_ss = pearsonr_std(second_rs, second_size)

    stat = np.arctanh(first_rs) - np.arctanh(second_rs)
    pvalue = None

    if (alternative == "less"):
        pvalue = scipy.stats.norm.cdf(stat,
            scale=np.sqrt(first_ss**2 + second_ss**2))
    elif (alternative == "greater"):
        pvalue = 1 - scipy.stats.norm.cdf(stat,
            scale=np.sqrt(first_ss**2 + second_ss**2))
    elif (alternative == "two-sided"):
        pvalue = 2 * scipy.stats.norm.cdf(-np.abs(stat),
            scale=np.sqrt(first_ss**2 + second_ss**2))

    stat[first_rs == second_rs] = 0
    pvalue[first_rs == second_rs] = 0

    return stat, pvalue
