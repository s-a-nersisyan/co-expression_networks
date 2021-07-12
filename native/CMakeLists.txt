cmake_minimum_required(VERSION 3.4)
project(PCorr)

add_compile_options(-O3)
add_compile_options(-Wall)
add_compile_options(-march=native)

add_subdirectory(external/pybind11)

pybind11_add_module(pearson_correlation src/pearson_correlation.cpp)
