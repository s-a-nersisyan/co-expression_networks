#!/bin/bash

mkdir build
cd build
cmake ../native
make
cp *.so ../core/fast_computations/
