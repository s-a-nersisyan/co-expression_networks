import numpy as np

from scipy.stats import t
from scipy.stats import spearmanr

from .utils import *

PARTITION_SIZE = 10**3

EPS1 = 1e-5
EPS2 = 1e-2
EPS3 = 1e-5


def spearmanr_std(rs, size):
    return np.sqrt((1 + rs**2 / 2) / (size - 3))

def spearmanr_stats(target, sample, stat="rs"):
    rs = spearmanr(target, sample)[0]
    if (stat== "rs"):
        return rs
    elif (stat == "std"):
        return spearmanr_std(rs, len(target))
    else:
        return rs, spearmanr_std(rs, len(target))

def spearmanr_proba(quantiles, rs, ss, size):
    return t.cdf(
        (np.arctanh(quantiles) - np.arctanh(rs)) / ss,
        size - 2
    )


def spermanr_diff_analytic_proba(
    first_rs, first_size,
    second_rs, second_size,
    delta
):
    # first_rs = np.array(first_rs)
    # second_rs = np.array(second_rs)
    # delta = np.array(delta)

    proba = np.zeros(len(delta))

    indexes_1 = np.logical_and(np.abs(first_rs) < 1 - EPS2,
        np.abs(second_rs) < 1 - EPS2)
    indexes_2 = np.logical_xor(np.abs(first_rs) >= 1 - EPS2,
        np.abs(second_rs) >= 1 - EPS2)
    indexes_3 = np.logical_and(np.abs(first_rs) >= 1 - EPS2,
        np.abs(second_rs) >= 1 - EPS2)

    if (np.sum(indexes_1) > 0):
        proba[indexes_1] = _spermanr_diff_analytic_proba_1(
            first_rs[indexes_1], first_size,
            second_rs[indexes_1], second_size,
            delta[indexes_1]
        )

    if (np.sum(indexes_2) > 0):
        proba[indexes_2] = _spermanr_diff_analytic_proba_2(
            first_rs[indexes_2], first_size,
            second_rs[indexes_2], second_size,
            delta[indexes_2]
        )

    if (np.sum(indexes_3) > 0):
        proba[indexes_3] = _spermanr_diff_analytic_proba_3(
            first_rs[indexes_3], first_size,
            second_rs[indexes_3], second_size,
            delta[indexes_3]
        )

    return proba

def _spermanr_diff_analytic_proba_1(
    first_rs, first_size,
    second_rs, second_size,
    delta
):
    # check that all correlations are not equal to 1
    assert(np.sum(np.logical_or(np.abs(first_rs) >= 1 - EPS2,
        np.abs(second_rs) >= 1 - EPS2)) == 0)

    delta = delta.reshape((-1, 1))

    quantiles = np.linspace(-1, 1, PARTITION_SIZE).reshape((1, -1))
    quantiles_pd = quantiles + delta
    quantiles_md = quantiles - delta

    quantiles = bound(quantiles, -1 + EPS1, 1 - EPS1)
    quantiles_pd = bound(quantiles_pd, -1 + EPS1, 1 - EPS1)
    quantiles_md = bound(quantiles_md, -1 + EPS1, 1 - EPS1)

    first_ss = spearmanr_std(first_rs, first_size)
    second_ss = spearmanr_std(second_rs, second_size)

    first_rs = first_rs.reshape((-1, 1))
    first_ss = first_ss.reshape((-1, 1))
    second_rs = second_rs.reshape((-1, 1))
    second_ss = second_ss.reshape((-1, 1))

    fd_pd = spearmanr_proba(quantiles_pd, first_rs, first_ss, first_size)
    fd_md = spearmanr_proba(quantiles_md, first_rs, first_ss, first_size)
    sd = spearmanr_proba(quantiles, second_rs, second_ss, second_size)

    return np.sum(
        (fd_pd[:, :-1] - fd_md[:, :-1]) * (sd[:, 1:] - sd[:, :-1]),
        axis=1
    )

def _spermanr_diff_analytic_proba_2(
    first_rs, first_size,
    second_rs, second_size,
    delta
):
    # check that all correlations are equal to 1
    assert(np.sum(np.logical_xor(np.abs(first_rs) >= 1 - EPS2,
        np.abs(second_rs) >= 1 - EPS2)) == len(delta))

    indexes = (np.abs(first_rs) >= 1 - EPS2)
    first_rs[indexes], second_rs[indexes] =\
        second_rs[indexes], first_rs[indexes]

    first_ss = spearmanr_std(first_rs, first_size)

    proba = np.zeros(len(delta))

    delta_indexes = (delta <= EPS3) | (delta >= 2 - EPS3)
    proba[delta <= EPS3] = 0
    proba[delta >= 2 - EPS3] = 1

    p_indexes = np.logical_and(
        not delta_indexes,
        second_rs >= 1 - EPS2
    )
    m_indexes = np.logical_and(
        not delta_indexes,
        second_rs <= -1 + EPS2
    )

    proba[p_indexes] = 1 - spearmanr_proba(
        1 - delta[p_indexes], first_rs[p_indexes],
        first_ss[p_indexes], first_size
    )

    proba[m_indexes] = spearmanr_proba(
        -1 + delta[m_indexes], first_rs[m_indexes],
        first_ss[m_indexes], first_size
    )

    return proba

def _spermanr_diff_analytic_proba_3(
    first_rs, first_size,
    second_rs, second_size,
    delta
):
    # check that all correlations are equal to 1
    assert(np.sum(np.logical_and(np.abs(first_rs) < 1 - EPS2,
        np.abs(second_rs) < 1 - EPS2)) == 0)

    proba = np.zeros(len(delta))
    proba[np.abs(first_rs - second_rs) <= delta] = 1
    return proba

def spermanr_diff_bootstrap_proba(
    first_rs_samples, second_rs_samples,
    delta
):
    delta = delta.reshape((-1, 1))
    proba = np.sum(
        np.abs(first_rs_samples - second_rs_samples) <= delta,
        axis=1
    ) / len(first_rs_samples[0])
    return proba

def spearmanr_diff_proba(
    first_target, first_sample,
    second_target, second_sample,
    delta, method="analytic"
):
    if (isinstance(delta, float)):
        delta = np.array([delta])
    else:
        delta = np.array(delta)

    first_sample = np.array(first_sample)
    second_sample = np.array(second_sample)

    proba = np.zeros(len(first_sample))

    if (method == "bootstrap"):
        first_rs_samples = []
        second_rs_samples = []
        for i in range(len(first_sample)):
            first_rs_samples.append([
                r for r, p in bootstrap_sample(
                    first_target,
                    first_sample[i],
                    statistic=spearmanr
                )
            ])

        for i in range(len(second_sample)):
            second_rs_samples.append([
                r for r, p in bootstrap_sample(
                    second_target,
                    second_sample[i],
                    statistic=spearmanr
                )
            ])

        first_rs_samples = np.array(first_rs_samples)
        second_rs_samples = np.array(second_rs_samples)

        return spermanr_diff_bootstrap_proba(
            first_rs_samples,
            second_rs_samples,
            delta
        )

    elif (method == "analytic"):
        first_rs = []
        for i in range(len(first_sample)):
            rs = spearmanr_stats(
                first_target, first_sample[i]
            )
            first_rs.append(rs)

        second_rs = []
        for i in range(len(second_sample)):
            rs = spearmanr_stats(
                second_target, second_sample[i]
            )
            second_rs.append(rs)


        first_rs = np.array(first_rs)
        second_rs = np.array(second_rs)

        return spermanr_diff_analytic_proba(
            first_rs, len(first_target),
            second_rs, len(second_target),
            delta
        )

    return None
