#ifndef UTILS_H
#define UTILS_H

#include <utility>
#include <vector>

const int UNDEFINED_INDEX = -1;


std::pair<int, int> paired_index(int index, int base);

int unary_index(int first, int second, int base);

std::vector<int> paired_vector(int index, int base);

#endif

