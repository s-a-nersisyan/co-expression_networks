#include <cmath>
#include <string>
#include <utility>

#include "utils.h"

int unary_index(int first, int second, int base) {
	/* This funcito computes an indexed of
	 * a pair ("first", "second") in the alphabetically
	 * ordered raw that was made by all unique pairs
	 * from "0" to "base" */

    if (first == second) {
        return UNDEFINED_INDEX;
    }
    
    if (first > second) {
        int tmp = second;
        second = first;
        first = tmp;
    }

    int unary_index = (2 * base - first - 1) * (first) / 2;
    unary_index += second - first - 1;

    return unary_index;
}

std::pair<int, int> paired_index(int index, int base) {
	/* This function is inverse to "unary_index" */

    int i = std::floor(((2 * base - 1) - std::sqrt((2 * base - 1) * 
            (2 * base - 1) - 8 * index)) / 2);
    int j = (index % base + ((i + 2) * (i + 1) / 2) % base) % base; 
    
	return std::pair<int, int>(i, j);
}

std::vector<int> unary_vector(int index, int base) {
	/* Computes all indexes in the alphabetically
	 * ordered raw with "index" in the paired index */

    std::vector<int> paired_array(base);
    for (int j = 0; j < base; ++j) {
        paired_array[j] = unary_index(index, j, base);    
    }
    
	return paired_array;
}
