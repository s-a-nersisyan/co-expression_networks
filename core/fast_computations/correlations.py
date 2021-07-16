import numpy as np
from scipy.stats import rankdata

from .correlation_computations import _pearsonr
from .correlation_computations import UNDEFINED_CORR_VALUE

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


def spearmanr(
    df,
    source_indexes, target_indexes,
    process_num=1,
    numerical_index=False
):
    if not numerical_index:
        source_num_indexes, target_num_indexes = _get_num_ind(
            df.index.to_list(),
            source_indexes, target_indexes
        )
     
    else:
        source_num_indexes = source_indexes
        target_num_indexes = target_indexes
 
    data = rankdata(
        df.to_numpy(copy=True),
        axis=1
    ).astype("float32")
    source_num_indexes = np.array(
        source_num_indexes
    ).astype("int32")
    target_num_indexes = np.array(
        target_num_indexes
    ).astype("int32")
 
    print("Correlation computations")
    corrs = _pearsonr(
        data,
        source_num_indexes,
        target_num_indexes,
        process_num
    )

    corrs[corrs == UNDEFINED_CORR_VALUE] = None
    return corrs



def pearsonr(
    df,
    source_indexes, target_indexes,
    process_num=1,
    numerical_index=False
):  
    if not numerical_index:
        source_num_indexes, target_num_indexes = _get_num_ind(
            df.index.to_list(),
            source_indexes, target_indexes
        )

    else:
        source_num_indexes = source_indexes
        target_num_indexes = target_indexes
        
    data = df.to_numpy(copy=True).astype("float32")
    source_num_indexes = np.array(
        source_num_indexes
    ).astype("int32")
    target_num_indexes = np.array(
        target_num_indexes
    ).astype("int32")
    
    print("Correlation computations")
    corrs = _pearsonr(
        data,
        source_num_indexes,
        target_num_indexes,
        process_num
    )

    corrs[corrs == UNDEFINED_CORR_VALUE] = None
    return corrs





