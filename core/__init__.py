# Add precompiled module for fast computations
# import sys
# sys.path.append("../build/")

# Fast correlation computations
from .fast_computations.correlations import \
    pearsonr, spearmanr


# Probabilistic utlis
from .correlation_utils import \
    spearmanr_mean, spearmanr_std, spearmanr_proba, \
    pearsonr_mean, pearsonr_std, pearsonr_proba, \
    correlation_diff_proba, correlation_diff_analytic_proba

# Test utils
from .correlation_tests import \
    corr_diff_test
