# Add precompiled module for fast computations
# import sys
# sys.path.append("../build/")

# Fast correlation computations
from .fast_computations.correlations import \
    pearsonr, spearmanr
from .fast_computations.tests import \
    corr_diff_test

# Probabilistic utlis
from .correlation_utils import \
    spearmanr_mean, spearmanr_std, spearmanr_cdf, \
    pearsonr_mean, pearsonr_std, pearsonr_cdf, \
    correlation_diff_cdf, correlation_diff_analytic_cdf

# Test utils
from .correlation_tests import \
    corr_diff_test as py_corr_diff_test

# Utils
from .utils import \
    paired_index


