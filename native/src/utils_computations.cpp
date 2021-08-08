#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cmath>
#include <iostream>
#include <thread>
#include <string>
#include <utility>
#include <queue>

#include "utils/utils.h"

namespace py = pybind11;

using NumPyFloatArray = py::array_t<float, py::array::c_style>;
using NumPyDoubleArray = py::array_t<double, py::array::c_style>;
using NumPyIntArray = py::array_t<int32_t, py::array::c_style>;


NumPyIntArray paired_array(const NumPyIntArray &index, int base) {
    py::buffer_info index_buf = index.request(); 
	int index_size = index_buf.shape[0];
	int *index_ptr = (int *) index_buf.ptr;

    NumPyIntArray paired_array = NumPyIntArray(index_size * base);
    int *pa_ptr = (int *) paired_array.request().ptr;
	
	for (int i = 0; i < index_size; ++i){
		for (int j = 0; j < base; ++j) {
			pa_ptr[i * base + j] = unary_index(index_ptr[i], j, base);	
		}
	}
	
	paired_array.resize({index_size, base});
	return paired_array;
}

NumPyFloatArray paired_reshape(
	const NumPyFloatArray &array,
	const NumPyIntArray &index,
	int base
) {

    py::buffer_info index_buf = index.request(); 
	int index_size = index_buf.shape[0];
	int *index_ptr = (int *) index_buf.ptr;
	
	float *array_ptr = (float *) array.request().ptr;

    NumPyFloatArray paired_array = NumPyFloatArray(index_size * base);
    float *pa_ptr = (float *) paired_array.request().ptr;
	
	for (int i = 0; i < index_size; ++i){
		for (int j = 0; j < base; ++j) {
			if (index_ptr[i] == j) {
				pa_ptr[i * base + j] = 1.;
				continue;
			}
				
			pa_ptr[i * base + j] = array_ptr[
				unary_index(index_ptr[i], j, base)
			];	
		}
	}
	
	paired_array.resize({index_size, base});
	return paired_array;
}

PYBIND11_MODULE(utils_computations, m) {
    m.def("_paired_index", &paired_index);
    m.def("_unary_index", &unary_index);
    m.def("_paired_array", &paired_array);
    m.def("_paired_reshape", &paired_reshape);
    m.attr("UNDEFINED_INDEX") = py::int_(UNDEFINED_INDEX);
}
