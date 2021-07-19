import numpy as np
import math


BOOTSTRAP_REPEATS = 10**3


def paired_index(index, base):
    i = (2 * base - 1) - np.sqrt((2 * base - 1)**2 - 8 * index)
    i /= 2
    i = math.floor(i)

    j = (index % base +  ((i + 2) * (i + 1) // 2) % base) % base 
    return i, j



def bound(array, left, right):
    array = np.array(array)
    array[array < left] = left
    array[array > right] = right
    return array

def bootstrap_sample(
    *args, statistic=None,
    bootstrap_repeats=BOOTSTRAP_REPEATS
):
    for i in range(BOOTSTRAP_REPEATS):
        indexes = np.random.choice(
            np.arange(len(args[0])),
            len(args[0]),
            replace=True
        )

        samples = []
        for arg in np.array(args):
            samples.append(arg[indexes])

        if (statistic != None):
            yield statistic(*samples)
        else:
            yield sample
