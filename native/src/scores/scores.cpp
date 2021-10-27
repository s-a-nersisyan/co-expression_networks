#include <cmath>
#include <string>
#include <utility>
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

        mean += data_ptr[index];
    }

    if (end_ind - start_ind > 0) {
        mean /= end_ind - start_ind;
    }

    return mean;
}

int _mean(
    float *data_ptr,
    int *starts_ind_ptr,
    int *ends_ind_ptr,
    float *scores_ptr,
    int start_ind,
    int end_ind
) {
    for (int i = start_ind; i < end_ind; ++i) {
        scores_ptr[start_ind + i] = mean(
            data_ptr,
            (int *) nullptr,
            starts_ind_ptr[i],
            ends_ind_ptr[i]
        );  
    }

    return 0;
}


int __mean(
    float *data_ptr,
    int sources_size,
    float *scores_ptr,
    int start_ind,
    int end_ind
) {
    for (int i = start_ind; i < end_ind; ++i) {
        std::vector<int> targets = unary_vector(i, sources_size);
        scores_ptr[i] = mean(
            data_ptr,
            targets.data(),
            0,
            sources_size
        );  
    }

    return 0;
}

float quantile(
    float *data_ptr,
    int *index_ptr,
    int start_ind,
    int end_ind,
    float q
) {
    if (end_ind - start_ind == 0) {
        return 0;
    }

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

    std::sort(values.begin(), values.end());
    
    return values[(int) (end_ind - start_ind) * q];
}

int _quantile(
    float *data_ptr,
    int *starts_ind_ptr,
    int *ends_ind_ptr,
    float *scores_ptr,
    int start_ind,
    int end_ind,
    float q
) {
    for (int i = start_ind; i < end_ind; ++i) {
        scores_ptr[i] = quantile(
            data_ptr,
            (int *) nullptr,
            starts_ind_ptr[i],
            ends_ind_ptr[i],
            q
        );  
    }

    return 0;
}

int __quantile(
    float *data_ptr,
    int sources_size,
    float *scores_ptr,
    int start_ind,
    int end_ind,
    float q
) {
    for (int i = start_ind; i < end_ind; ++i) {
        std::vector<int> targets = unary_vector(i, sources_size);
        scores_ptr[i] = quantile(
            data_ptr,
            targets.data(),
            0,
            sources_size,
            q
        );  
    }

    return 0;
}
