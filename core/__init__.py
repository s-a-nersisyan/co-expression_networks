# Add precompiled module for fast computations
# import sys
# sys.path.append("../build/")
from .correlation_computations import \
    pearsonr, UNDEFINED_CORR_VALUE


# Regular imports
from .correlation_utils import \
    spearmanr_mean, spearmanr_std, spearmanr_proba, \
    pearsonr_mean, pearsonr_std, pearsonr_proba, \
    correlation_diff_proba, correlation_diff_analytic_proba
from .correlation_tests import \
    pearsonr_test
