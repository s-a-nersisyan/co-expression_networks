import numpy as np
import math
import core.extern
import tqdm
import sys
import os
#from pandas import DataFrame

BOOTSTRAP_REPEATS = 10**3


def get_num_ind(indexes, *args):
    index_hash = {
        ind: num for num, ind in enumerate(indexes)
    }
    
    result = []
    for arg in args:
        result.append([
            index_hash[ind] for ind in arg
        ])
    
    if (len(result) == 1):
        return result[0]
    
    return result

def bound(array, left, right):
    array = np.array(array)
    array[array < left] = left
    array[array > right] = right
    return array

def bootstrap_sample(
    *args, statistic=None,
    bootstrap_repeats=BOOTSTRAP_REPEATS
):
    for i in range(bootstrap_repeats):
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


def checking_directory_existence(directory_path, message="Output directory does not exist!"):
    if os.path.isdir(directory_path) == False:
        print(message)
        sys.exit()


def saving_by_chunks(sort_ind, df_indexes, df_template, df_columns, 
                     path_to_file, n=10**6):
    df_inds = []
    rows = len(sort_ind)
    MODE, HEADER = 'w', True
    
    # Splitting rows into chunks
    for i in range(0, ((rows//n)+(rows%n))):
        if rows < (i+1)*n:
            df_inds.append([i*(n), rows])
            break
        df_inds.append([i*(n), (i+1)*n])
        
    for df_ind in tqdm.tqdm(df_inds, desc="Saving"):
        iter_indexes = sort_ind[df_ind[0]:df_ind[1]]
        
        source_indexes = []
        target_indexes = []
        for ind in (iter_indexes):
            s, t = core.extern.paired_index(ind, len(df_indexes))
            source_indexes.append(df_indexes[s])
            target_indexes.append(df_indexes[t])
        output_df = df_template.copy()
        output_df["Source"] = source_indexes
        output_df["Target"] = target_indexes
        for i in range(len(df_columns)):
            output_df.iloc[:,i+2] = df_columns[i][iter_indexes]

        output_df.to_csv(
            path_to_file,
            sep=",",
            index=None,
            mode=MODE,
            header=HEADER
        )
        MODE, HEADER = 'a', False

