#include <cmath>
#include <string>
#include <utility>
#include <queue>
#include <vector>
#include <algorithm>

#include "scores.h"
#include "../utils/utils.h"


float mean(
    float *data_ptr,
    int *index_ptr,
    int start_ind,
    int end_ind
) {
    if (end_ind - start_ind == 0) {
        return 0;
    }

    float mean = 0;

    int index = 0;
    for (int i = start_ind; i < end_ind; ++i) {
        if (!index_ptr) {
            index = i;
        } else {
            index = index_ptr[i];
        }

        mean += data_ptr[index_ptr[i]];
    }

    if (end_ind - start_ind > 0) {
        mean /= end_ind - start_ind;
    }

    return mean;
}

int _mean(
    float *data_ptr,
    float *start_ind_ptr,
    float *end_ind_ptr,
    float *scores_ptr,
    int start_ind,
    int end_ind
) {
    int size = end_ind - start_ind;
    for (int i = 0; i < size; ++i) {
        scores[start_ind + i] = mean(
            data_ptr,
            (int *) nullptr,
            starts_ind_ptr[i],
            ends_ind_ptr[i]
        );  
    }

    return 0;
}

float quantile(
    float *data_ptr,
    int *index_ptr,
    int start_ind,
    int end_ind,
    float quantile
) {
    if (end_ind - start_ind == 0) {
        return 0;
    }

    float median = 0;
    
    int index = 0;
    std::vector<float> values(end_ind - start_ind);
    for (int i = start_ind; i < end_ind; ++i) {
        if (!index_ptr) {
            index = i;
        } else {
            index = index_ptr[i];
        }

        values[i - start_ind] = data_ptr[index];
    }

    std::sort(values.begin(), values.edn());
    
    return values[(int) (end_ind - start_ind) * quantile];
}

int _quantile(
    float *data_ptr,
    float *start_ind_ptr,
    float *end_ind_ptr,
    float *scores_ptr,
    int start_ind,
    int end_ind,
    float quantile
) {
    int size = end_ind - start_ind;
    for (int i = 0; i < size; ++i) {
        scores[start_ind + i] = quantile(
            data_ptr,
            (int *) nullptr,
            starts_ind_ptr[i],
            ends_ind_ptr[i],
            quantile
        );  
    }

    return 0;
}
