import numpy as np

from .pipelines import \
    _ztest_pipeline_indexed, \
    _ztest_pipeline_exhaustive


def get_num_ind(indexes, *args):
    index_hash = {
        ind: num for num, ind in enumerate(indexes)
    }
    
    result = []
    for arg in args:
        result.append([
            index_hash[ind] for ind in arg
        ])

    return result 

def ztest_pipeline( 
    df,
    reference_indexes,
    experimental_indexes,
    source_indexes=None,
    target_indexes=None,
    correlation="spearman",
    alternative="two-sided",
    repeats_num=1000,
    process_num=1,
    numerical_index=False
):
    data = df.to_numpy(copy=True).astype("float32")

    if np.all(source_indexes) and np.all(target_indexes):
        if not numerical_index:
            source_num_indexes, target_num_indexes = \
                get_num_ind(
                    df.index.to_list(),
                    source_indexes,
                    target_indexes
                ) 
            ref_num_indexes, exp_num_indexes = \
                get_num_ind(
                    df.columns.to_list(),
                    reference_indexes,
                    experimental_indexes
                )
        else:
            source_num_indexes = source_indexes
            target_num_indexes = target_indexes
            
            ref_num_indexes = reference_indexes
            exp_num_indexes = experimental_indexes

        source_num_indexes = np.array(
            source_num_indexes
        ).astype("int32")
        target_num_indexes = np.array(
            target_num_indexes
        ).astype("int32")

        ref_num_indexes = np.array(
            ref_num_indexes
        ).astype("int32")
        exp_num_indexes = np.array(
            exp_num_indexes
        ).astype("int32")

        ref_corrs, exp_corrs, \
        stat, pvalue, bootstrap_pvalue = \
            _ztest_pipeline_indexed(
                data,
                source_num_indexes,
                target_num_indexes,
                ref_num_indexes,
                exp_num_indexes,
                correlation,
                alternative,
                repeats_num,
                process_num
            ) 
    else:
        if not numerical_index:
            ref_num_indexes, exp_num_indexes = \
                get_num_ind(
                    df.columns.to_list(),
                    reference_indexes,
                    experimental_indexes
                )
        else:
            ref_num_indexes = reference_indexes
            exp_num_indexes = experimental_indexes
        
        ref_num_indexes = np.array(
            ref_num_indexes
        ).astype("int32")
        exp_num_indexes = np.array(
            exp_num_indexes
        ).astype("int32")

        ref_corrs, exp_corrs, \
        stat, pvalue, bootstrap_pvalue = \
            _ztest_pipeline_exhaustive(
                data,
                ref_num_indexes,
                exp_num_indexes,
                correlation,
                alternative,
                repeats_num,
                process_num
            ) 

    return ref_corrs, exp_corrs, stat, pvalue, bootstrap_pvalue 
