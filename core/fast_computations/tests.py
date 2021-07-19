import numpy as np

from .correlation_computations import _corr_diff_test
from .correlation_computations import UNDEFINED_CORR_DIFF_TEST_VALUE

def corr_diff_test(
    first_rs, first_size,
    second_rs, second_size,
    correlation="spearman",
    alternative="two-seded",
    process_num=1 
):
    first_rs = np.array(first_rs, dtype="float32")
    first_size = np.array(first_size, dtype="int32")
    second_rs = np.array(second_rs, dtype="float32")
    second_size = np.array(second_size, dtype="int32")
    
    print("Test computations")
    stat, pvalue = _corr_diff_test(
        first_rs, first_size,
        second_rs, second_size,
        correlation,
        alternative,
        process_num
    )
    
    stat[pvalue == UNDEFINED_CORR_DIFF_TEST_VALUE] = None
    pvalue[pvalue == UNDEFINED_CORR_DIFF_TEST_VALUE] = None

    return stat, pvalue

