#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cmath>
#include <iostream>
#include <thread>
#include <string>
#include <utility>
#include <queue>

namespace py = pybind11;

using NumPyFloatArray = py::array_t<float, py::array::c_style>;
using NumPyDoubleArray = py::array_t<double, py::array::c_style>;
using NumPyIntArray = py::array_t<int32_t, py::array::c_style>;

const float UNDEFINED_CORR_VALUE = -2;
const float UNDEFINED_CORR_DIFF_TEST_VALUE = -2;

// Correlation block

int pearsonr_thread(
    float *data_ptr,
    int *source_ind_ptr,
    int *target_ind_ptr,
    float *corrs_ptr,
	int sample_size,
    int start_ind, int end_ind
) {
   
	for (int i = start_ind; i < end_ind; ++i) {
        int source_index = source_ind_ptr[i]; 
        int target_index = target_ind_ptr[i];

        float correlation = 0;
        float source_mean = 0, target_mean = 0;
        float source_var  = 0, target_var  = 0;

        for (int j = 0; j < sample_size; ++j) {
            correlation += data_ptr[source_index * sample_size + j] *
                data_ptr[target_index * sample_size + j];
            
            source_mean += data_ptr[source_index * sample_size + j];
            source_var  += data_ptr[source_index * sample_size + j] *
                           data_ptr[source_index * sample_size + j];
        
            target_mean += data_ptr[target_index * sample_size + j];
            target_var  += data_ptr[target_index * sample_size + j] *
                           data_ptr[target_index * sample_size + j]; 
        }

        source_mean /= sample_size;
        target_mean /= sample_size;

        correlation = correlation / sample_size - 
            source_mean * target_mean;

        source_var = source_var / sample_size -
            source_mean * source_mean; 
        target_var = target_var / sample_size -
            target_mean * target_mean; 
        
        if (source_var == 0 || target_var == 0) { 
            correlation = -2;
        } else {
            correlation /= std::sqrt(source_var * target_var);
        }

        corrs_ptr[i] = correlation;
    }

    return 0;
}

NumPyFloatArray pearsonr(
    const NumPyFloatArray &data,
    const NumPyIntArray &source_indexes,
    const NumPyIntArray &target_indexes,
    int process_num=1
) {    
    py::buffer_info source_ind_buf = source_indexes.request();
    py::buffer_info target_ind_buf = target_indexes.request();
    py::buffer_info data_buf = data.request();
    
	int sample_size = data_buf.shape[1]; 
	int index_size = source_ind_buf.shape[0];
    if (source_ind_buf.size != target_ind_buf.size) {
        throw std::runtime_error("Index shapes must match");
    }

    if (process_num > index_size) {
        process_num = index_size;
    }

    if (process_num <= 0) {	
        throw std::runtime_error("Process number error");
    }
    
    NumPyFloatArray corrs = NumPyFloatArray(source_ind_buf.size);
    float *corrs_ptr = (float *) corrs.request().ptr;
    
    if (process_num == 1) {
        pearsonr_thread(
            (float *) data_buf.ptr,
            (int *) source_ind_buf.ptr,
            (int *) target_ind_buf.ptr,
            corrs_ptr,
			sample_size,
            0, index_size
        );

    } else {
        std::queue<std::thread> threads;
        int batch_size = index_size / process_num;
        for (int i = 0; i < process_num; ++i) {
            int left_border = i * batch_size;
            int right_border = (i + 1) * batch_size;
            if (i == process_num - 1) {
                right_border = index_size;
            }
            
	    	std::thread thr(pearsonr_thread,
                (float *) data_buf.ptr,
                (int *) source_ind_buf.ptr,
                (int *) target_ind_buf.ptr,
                corrs_ptr,
				sample_size,
                left_border, right_border
            );
            
			threads.push(move(thr));
        }

		while (!threads.empty()) {
			threads.front().join();
			threads.pop();
		}
    }

    return corrs;
}

std::pair<int, int> paired_index(int index, int base) {
	int i = std::floor(((2 * base - 1) - std::sqrt((2 * base - 1) * 
			(2 * base - 1) - 8 * index)) / 2);
	int j = (index % base + ((i + 2) * (i + 1) / 2) % base) % base; 
	return std::pair<int, int>(i, j);
}

int pearsonr_unindexed_thread(
    float *data_ptr,
    float *corrs_ptr,
	int sample_size,
    int start_ind, int end_ind,
	int index_size
) {
   
	for (int i = start_ind; i < end_ind; ++i) {
		std::pair<int, int> paired_ind =
			paired_index(i, index_size);
        int source_index = paired_ind.first; 
        int target_index = paired_ind.second;

        float correlation = 0;
        float source_mean = 0, target_mean = 0;
        float source_var  = 0, target_var  = 0;

        for (int j = 0; j < sample_size; ++j) {
            correlation += data_ptr[source_index * sample_size + j] *
                data_ptr[target_index * sample_size + j];
            
            source_mean += data_ptr[source_index * sample_size + j];
            source_var  += data_ptr[source_index * sample_size + j] *
                           data_ptr[source_index * sample_size + j];
        
            target_mean += data_ptr[target_index * sample_size + j];
            target_var  += data_ptr[target_index * sample_size + j] *
                           data_ptr[target_index * sample_size + j]; 
        }

        source_mean /= sample_size;
        target_mean /= sample_size;

        correlation = correlation / sample_size - 
            source_mean * target_mean;

        source_var = source_var / sample_size -
            source_mean * source_mean; 
        target_var = target_var / sample_size -
            target_mean * target_mean; 
        
        if (source_var == 0 || target_var == 0) { 
            correlation = -2;
        } else {
            correlation /= std::sqrt(source_var * target_var);
        }

        corrs_ptr[i] = correlation;
    }

    return 0;
}

NumPyFloatArray pearsonr_unindexed(
    const NumPyFloatArray &data,
    int process_num=1
) {    
    py::buffer_info data_buf = data.request();
    
	int sample_size = data_buf.shape[1]; 
	int index_size = data_buf.shape[0];
	int pairs_num = index_size * (index_size - 1) / 2;

    if (process_num > index_size) {
        process_num = index_size;
    }

    if (process_num <= 0) {	
        throw std::runtime_error("Process number error");
    }
    
    NumPyFloatArray corrs = NumPyFloatArray(pairs_num);
    float *corrs_ptr = (float *) corrs.request().ptr;
    
    if (process_num == 1) {
        pearsonr_unindexed_thread(
            (float *) data_buf.ptr,
            corrs_ptr,
			sample_size,
            0, pairs_num,
			index_size
        );

    } else {
        std::queue<std::thread> threads;
        int batch_size = pairs_num / process_num;
        for (int i = 0; i < process_num; ++i) {
            int left_border = i * batch_size;
            int right_border = (i + 1) * batch_size;
            if (i == process_num - 1) {
                right_border = pairs_num;
            }
            
	    	std::thread thr(pearsonr_unindexed_thread,
                (float *) data_buf.ptr,
                corrs_ptr,
				sample_size,
                left_border, right_border,
				index_size
            );
            
			threads.push(move(thr));
        }

		while (!threads.empty()) {
			threads.front().join();
			threads.pop();
		}
    }

    return corrs;
}

// Test block

float norm_cdf(float x) {
	return std::erfc(-x / std::sqrt(2)) / 2;
}

int corr_diff_test_thread(
    float *first_rs_ptr, int *first_size_ptr,
    float *second_rs_ptr, int *second_size_ptr,
	float *stat_ptr, float *pvalue_ptr,
	int start_ind, int end_ind,
    std::string &correlation,
    std::string &alternative
) {

	for (int ind = start_ind; ind < end_ind; ++ind) {
		if (std::abs(first_rs_ptr[ind] + second_rs_ptr[ind]) == 2) {
			stat_ptr[ind] = UNDEFINED_CORR_DIFF_TEST_VALUE;
			pvalue_ptr[ind] = UNDEFINED_CORR_DIFF_TEST_VALUE;
			continue;
		}

		float stat = (std::atanh(first_rs_ptr[ind]) -
				std::atanh(second_rs_ptr[ind]));	

		// This block is a copy of 
		// core.correlation_utils.pearson_std
		float first_ss = 1 / std::sqrt(first_size_ptr[ind] - 3);
		float second_ss = 1 / std::sqrt(second_size_ptr[ind] - 3);
		if (correlation == "spearman") {
			first_ss *= std::sqrt(1.5);
			second_ss *= std::sqrt(1.5);
		}

		float std = std::sqrt(first_ss * first_ss +
				second_ss * second_ss);	
		
		float pvalue = UNDEFINED_CORR_DIFF_TEST_VALUE;
		if (alternative == "less") {
			pvalue = norm_cdf(stat * std);	
		} else if (alternative == "greater") {
			pvalue = 1 - norm_cdf(stat * std);
		} else if (alternative == "two-sided") {
			pvalue = 2 * norm_cdf(-std::abs(stat) * std);			
		}
		
		stat_ptr[ind] = stat;
		pvalue_ptr[ind] = pvalue;	
	}

	return 0;
}

std::pair<NumPyFloatArray, NumPyFloatArray> corr_diff_test(
    const NumPyFloatArray &first_rs,
    const NumPyIntArray &first_size,
    const NumPyFloatArray &second_rs,
    const NumPyIntArray &second_size,
    std::string correlation="spearman",
    std::string alternative="two-sided",
    int process_num=1
) { 
    
	py::buffer_info first_rs_buf = first_rs.request();
    py::buffer_info second_rs_buf = second_rs.request();
    
	int rs_number = first_rs_buf.shape[0];
    if (first_rs_buf.size != second_rs_buf.size) {
        throw std::runtime_error("Correlation shapes must match");
    }

    if (process_num > rs_number) {
        process_num = rs_number;
    }

    if (process_num <= 0) {	
        throw std::runtime_error("Process number error");
    }
    
    NumPyFloatArray stat = NumPyFloatArray(first_rs_buf.size);
    NumPyFloatArray pvalue = NumPyFloatArray(first_rs_buf.size);
    
    float *stat_ptr = (float *) stat.request().ptr;
    float *pvalue_ptr = (float *) pvalue.request().ptr;
    
    if (process_num == 1) {
        corr_diff_test_thread(
            (float *) first_rs_buf.ptr, (int *) first_size.request().ptr,
            (float *) second_rs_buf.ptr, (int *) second_size.request().ptr,
	    	stat_ptr, pvalue_ptr,
	    	0, rs_number,
            correlation,
            alternative
        );

    } else {
        std::queue<std::thread> threads;
        int batch_size = rs_number / process_num;
        for (int i = 0; i < process_num; ++i) {
            int left_border = i * batch_size;
            int right_border = (i + 1) * batch_size;
            if (i == process_num - 1) {
                right_border = rs_number;
            }
			
			std::thread thr(corr_diff_test_thread,
				(float *) first_rs_buf.ptr, (int *) first_size.request().ptr,
				(float *) second_rs_buf.ptr, (int *) second_size.request().ptr,
				stat_ptr, pvalue_ptr,
				left_border, right_border,
				std::ref(correlation),
				std::ref(alternative)
			);
				
			threads.push(move(thr));
        }

		while (!threads.empty()) {
			threads.front().join();
			threads.pop();
		}
    }

    return std::pair<NumPyFloatArray,
		   NumPyFloatArray>(stat, pvalue);
}

PYBIND11_MODULE(correlation_computations, m) {
    m.def("_pearsonr", &pearsonr);
    m.def("_pearsonr_unindexed", &pearsonr_unindexed);
    m.attr("UNDEFINED_CORR_VALUE") = py::float_(UNDEFINED_CORR_VALUE);
	m.def("_corr_diff_test", &corr_diff_test);
    m.attr("UNDEFINED_CORR_DIFF_TEST_VALUE") =
		py::float_(UNDEFINED_CORR_DIFF_TEST_VALUE);
}
