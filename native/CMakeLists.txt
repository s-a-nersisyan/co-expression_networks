cmake_minimum_required(VERSION 3.4)
project(correlation_computations)

add_compile_options(-O3)
add_compile_options(-Wall)
add_compile_options(-march=native)

add_subdirectory(extern/pybind11)

pybind11_add_module(correlation_computations src/correlation_computations.cpp)
