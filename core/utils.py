import numpy as np


BOOTSTRAP_REPEATS = 10**3


def bound(array, left, right):
    array = np.array(array)
    array[array < left] = left
    array[array > right] = right
    return array

def bootstrap_sample(*args, statistic=None):
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
