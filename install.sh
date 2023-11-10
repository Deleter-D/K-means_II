#!/bin/bash

if [ ! -d "./build/" ];then
    mkdir build
else
    echo "build folder is exist."
fi

cd build
cmake ..
make -j64
make install