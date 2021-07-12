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


NumPyFloatArray _pearson_correlation(
        const NumPyFloatArray &data,
        const NumPyFloatArray &source_indexes,
        const NumPyFloatArray &target_indexes,
        NumPyFloatArray &corrs,
        int start_ind, int end_ind) {
    
    py::buffer_info source_ind_buf = source_indexes.request();
    py::buffer_info target_ind_buf = target_indexes.request();
    py::buffer_info data_buf = data.request();
    
    py::buffer_info corrs_buf = corrs.request();

    float *data_ptr = (float *) data_buf.ptr;
    float *corrs_ptr = (float *) corrs_buf.ptr;
    int *source_ind_ptr = (int *) source_ind_buf.ptr;
    int *target_ind_ptr = (int *) target_ind_buf.ptr;
    
    int sample_size = data_buf.shape[1];
    for (int i = start_ind; i < end_ind; ++i) {
        int source_index = source_ind_ptr[i]; 
        int target_index = target_ind_ptr[i]; 
        
        float correlation = 0;
        float source_mean = 0,
               target_mean = 0;
        float source_var = 0,
               target_var = 0;

        for (int j = 0; j < sample_size; ++j) {
            correlation += data_ptr[source_index * sample_size + j] *
                data_ptr[target_index * sample_size + j];
            
            source_mean += data_ptr[source_index * sample_size + j];
            source_var += data_ptr[source_index * sample_size + j] *
                data_ptr[source_index * sample_size + j];
        
            target_mean += data_ptr[target_index * sample_size + j];
            target_var += data_ptr[target_index * sample_size + j] *
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
        
        correlation /= sqrt(source_var * target_var);
        corrs_ptr[i] = correlation;
    }
    
    return corrs;
}

NumPyFloatArray pearson_correlation(
        const NumPyFloatArray &data,
        const NumPyFloatArray &source_indexes,
        const NumPyFloatArray &target_indexes,
        int process_num) {
    
    py::buffer_info source_ind_buf = source_indexes.request();
    py::buffer_info target_ind_buf = target_indexes.request();
    py::buffer_info data_buf = data.request();
    
    if (source_ind_buf.size != target_ind_buf.size) {
        throw std::runtime_error("Index shapes must match");
    }
    
    NumPyFloatArray corrs = NumPyFloatArray(source_ind_buf.size);
    int index_size = data_buf.shape[0];
    
    if (process_num <= 1) {
        _pearson_correlation(
                data,
                source_indexes,
                target_indexes,
                corrs, 
                0, index_size
        );
    } else {
        std::vector<std::thread> threads;
        int batch_size = index_size / process_num;
        for (int i = 0; i < process_num - 1; ++i) {
            std::thread thr(_pearson_correlation,
                std::ref(data),
                std::ref(source_indexes),
                std::ref(target_indexes),
                std::ref(corrs), 
                i * batch_size, (i + 1) * batch_size
            );
            threads.push_back(move(thr));
        }

        std::thread thr(_pearson_correlation,
            std::ref(data),
            std::ref(source_indexes),
            std::ref(target_indexes),
            std::ref(corrs),
            (process_num - 1) * batch_size, index_size
        );
        threads.push_back(move(thr));

        for (size_t i = 0; i < threads.size(); ++i) {
            threads[i].join();
        }
    }

    return corrs;
}

PYBIND11_MODULE(correlation_src, m) {
    m.def("pcorr", &pearson_correlation);        
}
