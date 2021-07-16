#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cmath>
#include <iostream>
#include <thread>

namespace py = pybind11;

using NumPyFloatArray = py::array_t<float, py::array::c_style>;
using NumPyDoubleArray = py::array_t<double, py::array::c_style>;
using NumPyIntArray = py::array_t<int32_t, py::array::c_style>;

const float UNDEFINED_CORR_VALUE = -2;


int pearsonr_thread(
    const NumPyFloatArray &data,
    const NumPyIntArray &source_indexes,
    const NumPyIntArray &target_indexes,
    float *corrs_ptr,
    int start_ind, int end_ind
) {
    py::buffer_info source_ind_buf = source_indexes.request();
    py::buffer_info target_ind_buf = target_indexes.request();
    py::buffer_info data_buf = data.request();
    
    float *data_ptr = (float *) data_buf.ptr;
    int *source_ind_ptr = (int *) source_ind_buf.ptr;
    int *target_ind_ptr = (int *) target_ind_buf.ptr;
    
    int sample_size = data_buf.shape[1];
    for (int i = start_ind; i < end_ind; ++i) {
        int source_index = source_ind_ptr[i]; 
        int target_index = target_ind_ptr[i];

	if ((source_index >= data_buf.shape[0]) ||
	    (target_index >= data_buf.shape[0])) {
		throw std::runtime_error("Request index problem");
	}
        
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
            correlation /= sqrt(source_var * target_var);
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
            data,
            source_indexes,
            target_indexes,
            corrs_ptr, 
            0, index_size
        );

    } else {
        std::vector<std::thread> threads;
        int batch_size = index_size / process_num;
        for (int i = 0; i < process_num; ++i) {
            int left_border = i * batch_size;
            int right_border = (i + 1) * batch_size;
            if (i == process_num - 1) {
                right_border = index_size;
            }
            
	    std::thread thr(pearsonr_thread,
                std::ref(data),
                std::ref(source_indexes),
                std::ref(target_indexes),
                corrs_ptr, 
                left_border, right_border
            );
            threads.push_back(move(thr));
        }

        for (size_t i = 0; i < threads.size(); ++i) {
            threads[i].join();
        }
    }

    return corrs;
}

PYBIND11_MODULE(correlation_computations, m) {
    m.def("_pearsonr", &pearsonr);
    m.attr("UNDEFINED_CORR_VALUE") = py::float_(UNDEFINED_CORR_VALUE);
}
