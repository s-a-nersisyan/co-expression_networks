import numpy as np

from .utils_computations import \
        _paired_index, \
        _unary_index, \
        _paired_array, \
        _paired_reshape

from .utils_computations import UNDEFINED_INDEX


def paired_index(index, base):
    return _paired_index(index, base)

def unary_index(first, second, base):
    result = _unary_index(first, second, base)
    
    if result == UNDEFINED_INDEX:
        return None
    
    return result

def paired_array(index, base):
    index = np.array(index, dtype="int32")
    
    return _paired_array(index, base)

def paired_reshape(array, index, base):
    index = np.array(index, dtype="int32")
    array = np.array(array, dtype="float32")
    
    return _paired_reshape(array, index, base)

