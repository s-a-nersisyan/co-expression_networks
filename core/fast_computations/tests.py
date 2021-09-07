import numpy as np

from .correlation_computations import _corr_diff_test
from .correlation_computations import UNDEFINED_CORR_DIFF_TEST_VALUE

LEFT_BORDER_BOUND = -1 + 1.e-6
RIGHT_BORDER_BOUND = 1 - 1.e-6


# This function is similar to the
# one placed in utils.py
def bound(array, left, right):
    array = np.array(array)
    array[array < left] = left
    array[array > right] = right
    return array

def corr_diff_test(
    first_rs, first_size,
    second_rs, second_size,
    correlation="spearman",
    alternative="two-sided",
    process_num=1 
):
    first_rs = np.array(
        bound(first_rs, LEFT_BORDER_BOUND, RIGHT_BORDER_BOUND),
        dtype="float32"
    )
    first_size = np.array(first_size, dtype="int32")
    
    second_rs = np.array(
        bound(second_rs, LEFT_BORDER_BOUND, RIGHT_BORDER_BOUND),
        dtype="float32"
    )
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
