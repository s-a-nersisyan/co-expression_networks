cmake_minimum_required(VERSION 3.4)
project(CorrUtils)

add_compile_options(-O3)
add_compile_options(-Wall)
add_compile_options(-march=native)

add_subdirectory(extern/pybind11)

pybind11_add_module(CorrUtils src/corr_utils.cpp)
