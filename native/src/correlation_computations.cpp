#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cmath>
#include <iostream>
#include <thread>
#include <string>
#include <utility>
#include <queue>
#include <tuple>
#include <map>

#include <algorithm>
#include <random>
#include <vector>

#include "utils/utils.h"

namespace py = pybind11;

using NumPyFloatArray = py::array_t<float, py::array::c_style>;
using NumPyDoubleArray = py::array_t<double, py::array::c_style>;
using NumPyIntArray = py::array_t<int32_t, py::array::c_style>;

const float UNDEFINED_CORR_VALUE = -2;
const float UNDEFINED_CORR_DIFF_TEST_VALUE = -2;

const float LEFT_CORR_BOUND = -0.99;
const float RIGHT_CORR_BOUND = 0.99;

const int REPEATS_NUMBER = 1000;

// Correlation block

int pearsonr_thread(
    float *data_ptr,
    int *source_ind_ptr,
    int *target_ind_ptr,
    float *corrs_ptr,
    int sample_size,
    int start_ind, int end_ind,
	int *sample_ind_ptr=nullptr
) {
    for (int i = start_ind; i < end_ind; ++i) {
        int source_index = source_ind_ptr[i]; 
        int target_index = target_ind_ptr[i];

        float correlation = 0;
        float source_mean = 0, target_mean = 0;
        float source_var  = 0, target_var  = 0;

        for (int j = 0; j < sample_size; ++j) {
			int jj = (sample_ind_ptr == nullptr) ? j : sample_ind_ptr[j];
            correlation += data_ptr[source_index * sample_size + jj] *
                data_ptr[target_index * sample_size + jj];
            
            source_mean += data_ptr[source_index * sample_size + jj];
            source_var  += data_ptr[source_index * sample_size + jj] *
                           data_ptr[source_index * sample_size + jj];
        
            target_mean += data_ptr[target_index * sample_size + jj];
            target_var  += data_ptr[target_index * sample_size + jj] *
                           data_ptr[target_index * sample_size + jj]; 
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
            correlation = UNDEFINED_CORR_VALUE;
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
                left_border, right_border,
				nullptr
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

int pearsonr_unindexed_thread(
    float *data_ptr,
    float *corrs_ptr,
    int sample_size,
    int start_ind, int end_ind,
    int index_size,
	int *sample_ind_ptr=nullptr
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
			int jj = (sample_ind_ptr == nullptr) ? j : sample_ind_ptr[j];
            correlation += data_ptr[source_index * sample_size + jj] *
                		   data_ptr[target_index * sample_size + jj];
            
            source_mean += data_ptr[source_index * sample_size + jj];
            source_var  += data_ptr[source_index * sample_size + jj] *
                           data_ptr[source_index * sample_size + jj];
        
            target_mean += data_ptr[target_index * sample_size + jj];
            target_var  += data_ptr[target_index * sample_size + jj] *
                           data_ptr[target_index * sample_size + jj]; 
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
                index_size,
				nullptr
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

int corr_diff_test_unsized_thread(
    float *first_rs_ptr, int first_size,
    float *second_rs_ptr, int second_size,
    float *stat_ptr, float *pvalue_ptr,
    int start_ind, int end_ind,
    const std::string &correlation,
    const std::string &alternative
) {

    for (int ind = start_ind; ind < end_ind; ++ind) {
		float first_rs = first_rs_ptr[ind];
		float second_rs = second_rs_ptr[ind];
		
		// if (std::abs(first_rs_ptr[ind] + second_rs_ptr[ind]) == 2) {
        //     stat_ptr[ind] = UNDEFINED_CORR_DIFF_TEST_VALUE;
        //     pvalue_ptr[ind] = UNDEFINED_CORR_DIFF_TEST_VALUE;
        //     continue;
        // }
		
		// Bound corrs
		if (first_rs < LEFT_CORR_BOUND) {
			first_rs = LEFT_CORR_BOUND;
		}

		if (second_rs < LEFT_CORR_BOUND) {
			second_rs = LEFT_CORR_BOUND;
		}
		
		if (first_rs > RIGHT_CORR_BOUND) {
			first_rs = RIGHT_CORR_BOUND;
		}

		if (second_rs > RIGHT_CORR_BOUND) {
			second_rs = RIGHT_CORR_BOUND;
		}

        float stat = (std::atanh(first_rs) -
        		std::atanh(second_rs));    
		
        // This block is a copy of 
        // core.correlation_utils.pearson_std
        float first_ss = 1 / std::sqrt(first_size - 3);
        float second_ss = 1 / std::sqrt(second_size - 3);
        if (correlation == "spearman") {
            first_ss *= std::sqrt(1.5);
            second_ss *= std::sqrt(1.5);
        }

        // float std = std::sqrt(first_ss * first_ss +
        //         second_ss * second_ss);    
    	
		float std = 1;

		if (pvalue_ptr != nullptr) {
			float pvalue = UNDEFINED_CORR_DIFF_TEST_VALUE;
			if (alternative == "less") {
				pvalue = norm_cdf(stat / std);    
			} else if (alternative == "greater") {
				pvalue = 1 - norm_cdf(stat / std);
			} else if (alternative == "two-sided") {
				pvalue = 2 * norm_cdf(-std::abs(stat) / std);            
			}
			
        	pvalue_ptr[ind] = pvalue;
		}
   
        stat_ptr[ind] = stat;
    }

    return 0;
}

int corr_diff_test_sized_thread(
    float *first_rs_ptr, int *first_size_ptr,
    float *second_rs_ptr, int *second_size_ptr,
    float *stat_ptr, float *pvalue_ptr,
    int start_ind, int end_ind,
    const std::string &correlation,
    const std::string &alternative
) {
    for (int ind = start_ind; ind < end_ind; ++ind) {
  		float first_rs = first_rs_ptr[ind]; 
  		float second_rs = second_rs_ptr[ind]; 
		
		// if (std::abs(first_rs_ptr[ind] + second_rs_ptr[ind]) == 2) {
        //     stat_ptr[ind] = UNDEFINED_CORR_DIFF_TEST_VALUE;
        //     pvalue_ptr[ind] = UNDEFINED_CORR_DIFF_TEST_VALUE;
        //     continue;
        // }
		
		// Bound corrs
		if (first_rs < LEFT_CORR_BOUND) {
			first_rs = LEFT_CORR_BOUND;
		}

		if (second_rs < LEFT_CORR_BOUND) {
			second_rs = LEFT_CORR_BOUND;
		}
		
		if (first_rs > RIGHT_CORR_BOUND) {
			first_rs = RIGHT_CORR_BOUND;
		}

		if (second_rs > RIGHT_CORR_BOUND) {
			second_rs = RIGHT_CORR_BOUND;
		}

        float stat = (std::atanh(first_rs) -
                std::atanh(second_rs));    

        // This block is a copy of 
        // core.correlation_utils.pearson_std
        float first_ss = 1 / std::sqrt(first_size_ptr[ind] - 3);
        float second_ss = 1 / std::sqrt(second_size_ptr[ind] - 3);
        if (correlation == "spearman") {
            first_ss *= std::sqrt(1.5);
            second_ss *= std::sqrt(1.5);
        }

        // float std = std::sqrt(first_ss * first_ss +
        //        second_ss * second_ss);    
		
       	float std = 1;

		if (pvalue_ptr != nullptr) {
			float pvalue = UNDEFINED_CORR_DIFF_TEST_VALUE;
			if (alternative == "less") {
				pvalue = norm_cdf(stat / std);    
			} else if (alternative == "greater") {
				pvalue = 1 - norm_cdf(stat / std);
			} else if (alternative == "two-sided") {
				pvalue = 2 * norm_cdf(-std::abs(stat) / std);            
			}
			
        	pvalue_ptr[ind] = pvalue;
		}

        stat_ptr[ind] = stat;
    }

    return 0;
}

std::pair<NumPyFloatArray, NumPyFloatArray> corr_diff_test(
    const NumPyFloatArray &first_rs,
    const NumPyIntArray &first_size,
    const NumPyFloatArray &second_rs,
    const NumPyIntArray &second_size,
    const std::string correlation="spearman",
    const std::string alternative="two-sided",
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

	std::queue<std::thread> threads;
	int batch_size = rs_number / process_num;
	for (int i = 0; i < process_num; ++i) {
		int left_border = i * batch_size;
		int right_border = (i + 1) * batch_size;
		if (i == process_num - 1) {
			right_border = rs_number;
		}
		
		std::thread thr(corr_diff_test_sized_thread,
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

    return std::pair<NumPyFloatArray,
           NumPyFloatArray>(stat, pvalue);
}

// Pipeline block

int pipeline_thread(
	float *data_ptr,
	int *source_ind_ptr,
	int *target_ind_ptr,
	int start_ind, int end_ind,
	int *ref_ind_ptr,
	int *exp_ind_ptr,
	int ref_ind_size,
	int exp_ind_size,
	float *ref_corrs_ptr,
	float *exp_corrs_ptr,
	float *stat_ptr,
	float *pvalue_ptr,
	const std::string &correlation,
	const std::string &alternative
) {
	pearsonr_thread(
		data_ptr,
		source_ind_ptr,
		target_ind_ptr,
		ref_corrs_ptr,
		ref_ind_size,
		start_ind, end_ind,
		ref_ind_ptr
	);

	pearsonr_thread(
		data_ptr,
		source_ind_ptr,
		target_ind_ptr,
		exp_corrs_ptr,
		exp_ind_size,
		start_ind, end_ind,
		exp_ind_ptr
	);

	corr_diff_test_unsized_thread(
		ref_corrs_ptr, ref_ind_size,
		exp_corrs_ptr, exp_ind_size,
		stat_ptr, pvalue_ptr,
		start_ind, end_ind,
		correlation,
		alternative	
	);
	
	return 0;
}

std::tuple<NumPyFloatArray, NumPyFloatArray,
NumPyFloatArray, NumPyFloatArray, NumPyFloatArray>
corr_diff_test_boot(
	const NumPyFloatArray &data,
    const NumPyIntArray &source_indexes,
    const NumPyIntArray &target_indexes,
	const NumPyIntArray &reference_indexes,
	const NumPyIntArray &experimental_indexes,
	const std::string correlation,
	const std::string alternative,
	int repeats_number=REPEATS_NUMBER,
    int process_num=1	
) {
	py::buffer_info data_buf = data.request();
	float *data_ptr = (float *) data_buf.ptr;
	int sample_size = data_buf.shape[1];

	py::buffer_info source_ind_buf = source_indexes.request();
	int *source_ind_ptr = (int *) source_ind_buf.ptr;
	
	py::buffer_info target_ind_buf = target_indexes.request();
	int *target_ind_ptr = (int *) target_ind_buf.ptr;
	
	int index_size = source_ind_buf.shape[0];
	
	py::buffer_info ref_ind_buf = reference_indexes.request();
	int ref_ind_size = ref_ind_buf.shape[0];
	int *ref_ind_ptr = (int *) ref_ind_buf.ptr;
	
	py::buffer_info exp_ind_buf = experimental_indexes.request();
	int exp_ind_size = exp_ind_buf.shape[0];
	int *exp_ind_ptr = (int *) exp_ind_buf.ptr;

	// Real data
	// float *ref_corrs_ptr = new float[index_size];
	// float *exp_corrs_ptr = new float[index_size];
    NumPyFloatArray ref_corrs = NumPyFloatArray(index_size);
	float *ref_corrs_ptr = (float *) ref_corrs.request().ptr;
    
	NumPyFloatArray exp_corrs = NumPyFloatArray(index_size);
	float *exp_corrs_ptr = (float *) exp_corrs.request().ptr;

    NumPyFloatArray stat = NumPyFloatArray(index_size);
	float *stat_ptr = (float *) stat.request().ptr;
	
	NumPyFloatArray pvalue = NumPyFloatArray(index_size);
	float *pvalue_ptr = (float *) pvalue.request().ptr;

	NumPyFloatArray boot_pvalue = NumPyFloatArray(index_size);
	float *boot_pvalue_ptr = (float *) boot_pvalue.request().ptr;
	for (int i = 0; i < index_size; ++i) {
		boot_pvalue_ptr[i] = 0;
	}

	// Bootstrapped data
	float *boot_ref_corrs_ptr =  new float[index_size];
	float *boot_exp_corrs_ptr = new float[index_size];
	float *boot_stat_ptr = new float[index_size];
	 
	int *boot_ref_ind_ptr = new int[ref_ind_size];
	int *boot_exp_ind_ptr = new int[exp_ind_size];

	// Bootstrap indexes initialization
	std::vector<int> indexes(sample_size);
	for (int i = 0; i < sample_size; ++i) {
		indexes[i] = i;
	}

	// Random generator initialization
    std::random_device random_dev;
	std::mt19937 random_gen(random_dev());

	// Bootstrap pvalue computations	
	float *rcp, *ecp, *sp, *pv;
	int *rip, *eip;
	for (int r = 0; r < repeats_number + 1; ++r) {
		if (r == 0) {
			rcp = ref_corrs_ptr;
			ecp = exp_corrs_ptr;
			sp  = stat_ptr;
			pv = pvalue_ptr;

			rip = ref_ind_ptr;
			eip = exp_ind_ptr;
		} else {
			std::shuffle(indexes.begin(), indexes.end(), random_gen);
			for (int i = 0; i < ref_ind_size; ++i) {
				boot_ref_ind_ptr[i] = indexes[i];
			}
			for (int i = 0; i < exp_ind_size; ++i) {
				boot_exp_ind_ptr[i] = indexes[ref_ind_size + i];
			}

			rcp = boot_ref_corrs_ptr;
			ecp = boot_exp_corrs_ptr;
			sp  = boot_stat_ptr;
			pv  = nullptr;
			
			rip = boot_ref_ind_ptr;
			eip = boot_exp_ind_ptr;
		}
		
		std::queue<std::thread> threads;
		int batch_size = index_size / process_num;
		for (int i = 0; i < process_num; ++i) {
			int left_border = i * batch_size;
			int right_border = (i + 1) * batch_size;
			if (i == process_num - 1) {
				right_border = index_size;
			}
			
			std::thread thr(pipeline_thread,
				data_ptr,
				source_ind_ptr,
				target_ind_ptr,
				left_border, right_border,
				rip, eip,
				ref_ind_size,
				exp_ind_size,
				rcp, ecp, sp, pv,
				correlation,
				alternative
			);
			
			threads.push(move(thr));
		}

		while (!threads.empty()) {
			threads.front().join();
			threads.pop();
		}
		
		if (r > 0) {
			for (int i = 0; i < index_size; ++i) {
				if ((alternative == "two-sided") &&
						(std::abs(stat_ptr[i]) <= std::abs(boot_stat_ptr[i]))) {
					boot_pvalue_ptr[i] += 1;
				}

				if ((alternative == "less") &&
						(stat_ptr[i] <= boot_stat_ptr[i])) {
					boot_pvalue_ptr[i] += 1;
				}
				
				if ((alternative == "greater") &&
						(stat_ptr[i] >= boot_stat_ptr[i])) {
					boot_pvalue_ptr[i] += 1;
				}
			}
		}
	}
	
	for (int i = 0; i < index_size; ++i) {
		boot_pvalue_ptr[i] /= repeats_number;
	}

	// delete[] ref_corrs_ptr;
	// delete[] exp_corrs_ptr;
	
	delete[] boot_ref_corrs_ptr;
	delete[] boot_exp_corrs_ptr;
	delete[] boot_stat_ptr;
	 
	delete[] boot_ref_ind_ptr;
	delete[] boot_exp_ind_ptr;
	
	return std::tuple<
		NumPyFloatArray, NumPyFloatArray,
		NumPyFloatArray, NumPyFloatArray, NumPyFloatArray
	>(
		ref_corrs, exp_corrs, stat, pvalue, boot_pvalue
	);
}

// Score block

std::map<int, float> aggregate_scores(
	float *score_ptr,	
    int *index_ptr,
	int size
) {	
	std::map<int, float> agg_scores;
	std::map<int, int> scores_number;
	
	for (int i = 0; i < size; ++i) {
		if (!agg_scores.count(index_ptr[i])) {
			agg_scores[index_ptr[i]] = 0;
			scores_number[index_ptr[i]] = 0;
		}

		agg_scores[index_ptr[i]] +=
			score_ptr[i] * score_ptr[i];
		scores_number[index_ptr[i]] += 1;
	}

	for (int i = 0; i < size; ++i) {
		agg_scores[index_ptr[i]] /=
			scores_number[index_ptr[i]];
		agg_scores[index_ptr[i]] = std::sqrt(
			agg_scores[index_ptr[i]]
		);	
	}

	return agg_scores;
}

std::map<int, float> aggregate_scores(
	const NumPyFloatArray &scores,	
    const NumPyIntArray &indexes
) {
    py::buffer_info scr_buf = scores.request();
    py::buffer_info ind_buf = indexes.request();
	
	float *score_ptr = (float *) scr_buf.ptr;
	int *index_ptr = (int *) ind_buf.ptr;
    int size = ind_buf.shape[0];
	
	return aggregate_scores(
		score_ptr, index_ptr, size
	);
}

std::pair<std::map<int, float>, std::map<int, float>> score_pipeline(
	const NumPyFloatArray &data,
    const NumPyIntArray &source_indexes,
    const NumPyIntArray &target_indexes,
	const NumPyIntArray &reference_indexes,
	const NumPyIntArray &experimental_indexes,
	const std::string correlation,
	const std::string alternative,
	int repeats_number=REPEATS_NUMBER,
    int process_num=1	
) {
	py::buffer_info data_buf = data.request();
	float *data_ptr = (float *) data_buf.ptr;
	// int sample_size = data_buf.shape[1];

	py::buffer_info source_ind_buf = source_indexes.request();
	int *source_ind_ptr = (int *) source_ind_buf.ptr;
	
	py::buffer_info target_ind_buf = target_indexes.request();
	int *target_ind_ptr = (int *) target_ind_buf.ptr;
	
	int index_size = source_ind_buf.shape[0];
	
	py::buffer_info ref_ind_buf = reference_indexes.request();
	int ref_ind_size = ref_ind_buf.shape[0];
	int *ref_ind_ptr = (int *) ref_ind_buf.ptr;
	
	py::buffer_info exp_ind_buf = experimental_indexes.request();
	int exp_ind_size = exp_ind_buf.shape[0];
	int *exp_ind_ptr = (int *) exp_ind_buf.ptr;

	// Real data	
    NumPyFloatArray ref_corrs = NumPyFloatArray(index_size);
	float *ref_corrs_ptr = (float *) ref_corrs.request().ptr;

    NumPyFloatArray exp_corrs = NumPyFloatArray(index_size);
	float *exp_corrs_ptr = (float *) exp_corrs.request().ptr;

    NumPyFloatArray stat = NumPyFloatArray(index_size);
	float *stat_ptr = (float *) stat.request().ptr;
	
	// Bootstrapped data
	// float *boot_ref_corrs_ptr =  new float[index_size];
	// float *boot_exp_corrs_ptr = new float[index_size];
	// float *boot_stat_ptr = new float[index_size];
	// 
	// int *boot_ref_ind_ptr = new int[ref_ind_size];
	// int *boot_exp_ind_ptr = new int[exp_ind_size];

	// Score/pvalue computations
	std::map<int, float> scores;
	std::map<int, float> pvalues;
	
	std::map<int, float> boot_scores;
	
	float *rcp, *ecp, *sp;
	int *rip, *eip;
	for (int i = 0; i < repeats_number + 1; ++i) {
		if (i == 0) {
			rcp = ref_corrs_ptr;
			ecp = exp_corrs_ptr;
			sp  = stat_ptr;

			rip = ref_ind_ptr;
			eip = exp_ind_ptr;
		} else {
			// rcp = boot_ref_corrs_ptr;
			// ecp = boot_exp_corrs_ptr;
			// sp  = boot_stat_ptr;

			// rip = boot_ref_ind_ptr;
			// eip = boot_exp_ind_ptr;

			rcp = ref_corrs_ptr;
			ecp = exp_corrs_ptr;
			sp  = stat_ptr;

			rip = ref_ind_ptr;
			eip = exp_ind_ptr;

		}
		
		std::queue<std::thread> threads;
		int batch_size = index_size / process_num;
		for (int i = 0; i < process_num; ++i) {
			int left_border = i * batch_size;
			int right_border = (i + 1) * batch_size;
			if (i == process_num - 1) {
				right_border = index_size;
			}
			
			std::thread thr(pipeline_thread,
				data_ptr,
				source_ind_ptr,
				target_ind_ptr,
				left_border, right_border,
				rip, eip,
				ref_ind_size,
				exp_ind_size,
				rcp, ecp,
				sp, nullptr,
				correlation,
				alternative
			);
			
			threads.push(move(thr));
		}

		while (!threads.empty()) {
			threads.front().join();
			threads.pop();
		}
		
		if (i == 0) {
			scores = aggregate_scores(
				sp, source_ind_ptr, index_size
			);

			for (const auto &m : scores) {
				pvalues[m.first] = 0;  	
			}
		} else {
			boot_scores = aggregate_scores(
				sp, source_ind_ptr, index_size
			);

			for (const auto &m : boot_scores) {
				if (m.second >= scores[m.first]) {
					pvalues[m.first] += 1;
				}
			}
		}
	}

	for (const auto &m : pvalues) {
		pvalues[m.first] /= repeats_number;
	}
	
	return std::pair<
		std::map<int, float>,
		std::map<int, float>
	>(scores, pvalues);
}

PYBIND11_MODULE(correlation_computations, m) {
    m.def("_pearsonr", &pearsonr);
    m.def("_pearsonr_unindexed", &pearsonr_unindexed);
    m.attr("UNDEFINED_CORR_VALUE") = py::float_(UNDEFINED_CORR_VALUE);
    m.def("_corr_diff_test", &corr_diff_test);
	m.def("_corr_diff_test_boot", &corr_diff_test_boot);
    m.attr("UNDEFINED_CORR_DIFF_TEST_VALUE") =
        py::float_(UNDEFINED_CORR_DIFF_TEST_VALUE);
}
