cmake_minimum_required(VERSION 3.4)
project(fast_computations)

add_compile_options(-O3)
add_compile_options(-Wall)
add_compile_options(-march=native)

add_subdirectory(extern/pybind11)

# Can be used to compile lib
# add_subdirectory(src/utils)

pybind11_add_module(
	correlation_computations SHARED
	src/correlation_computations.cpp
	src/utils/utils.cpp
)

# Can be used to compile lib
# add_subdirectory(src/utils)
# target_link_libraries(correlation_computations PRIVATE utils)

pybind11_add_module(
	utils_computations SHARED
	src/utils_computations.cpp
	src/utils/utils.cpp
)
