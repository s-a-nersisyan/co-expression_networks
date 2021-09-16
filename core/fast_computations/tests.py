import numpy as np

from scipy.stats import rankdata

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

# This function is a copy of
# the one placed in ./correlations.py
def _get_num_ind(indexes, *args):
    index_hash = {
        ind: num for num, ind in enumerate(indexes)
    }
    
    result = []
    for arg in args:
        result.append([
            index_hash[ind] for ind in arg
        ])

    return result 


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
	df,
    source_indexes,
    target_indexes,
	reference_indexes,
	experimental_indexes,
    correlation="spearman",
    alternative="two-sided",
    repeats_num=1e3,
    process_num=1,
    numerical_index=False
):
    data = df.to_numpy(copy=True).astype("float32")
    if not numerical_index:
        source_num_indexes, target_num_indexes = _get_num_ind(
            df.index.to_list(),
            source_indexes, target_indexes
        )

        reference_num_indexes, experimental_num_indexes = _get_num_ind(
            df.columns.to_list(),
            reference_indexes, experimental_indexes
        )   
    else:
        source_num_indexes = source_indexes
        target_num_indexes = target_indexes
        reference_num_indexes = reference_indexes
        experimental_num_indexes = experimental_indexes
    
    source_num_indexes = np.array(
        source_num_indexes
    ).astype("int32")
    target_num_indexes = np.array(
        target_num_indexes
    ).astype("int32")

    reference_num_indexes = np.array(
        reference_num_indexes
    ).astype("int32")
    experimental_num_indexes = np.array(
        experimental_num_indexes
    ).astype("int32")

    indexes = set(source_num_indexes)
    indexes.update(target_num_indexes)
    indexes = list(indexes)

    # sample_indexes = set(reference_num_indexes)
    # sample_indexes.update(experimental_num_indexes)
    # sample_indexes = list(sample_indexes)

    if (correlation == "spearman"):
        data[indexes] = rankdata(
            data[indexes],
            axis=1
        ).astype("float32")

    print("Test bootstrap computations")
    ref_corrs, exp_corrs, \
    stat, pvalue, boot_pvalue = \
    _corr_diff_test_boot(
        data,
        source_num_indexes,
        target_num_indexes,
        reference_num_indexes,
        experimental_num_indexes,
        correlation,
        alternative,
        repeats_num,
        process_num
    )
    
    return ref_corrs, exp_corrs, stat, pvalue, boot_pvalue
