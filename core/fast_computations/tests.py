import numpy as np

from .correlation_computations import _corr_diff_test
from .correlation_computations import _corr_diff_test_boot
from .correlation_computations import UNDEFINED_CORR_DIFF_TEST_VALUE

LEFT_BORDER_BOUND = -0.99
RIGHT_BORDER_BOUND = 0.99


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

def corr_diff_test_boot(
	data,
    source_indexes,
    target_indexes,
	reference_indexes,
	experimental_indexes,
    correlation="spearman",
    alternative="two-sided",
    repeats_number=1000,
    process_num=1 
):
    data = np.array(data, dtype="float32")
    
    source_indexes = np.array(source_indexes, dtype="int32")
    target_indexes = np.array(target_indexes, dtype="int32")
    
    reference_indexes = np.array(reference_indexes, dtype="int32")
    experimental_indexes = np.array(experimental_indexes, dtype="int32")
    
    print("Test computations")
    stat, pvalue = _corr_diff_test_boot(
        first_rs, first_size,
        second_rs, second_size,
        correlation,
        alternative,
        process_num
    )
    
    return stat, pvalue

