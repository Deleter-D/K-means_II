cmake_minimum_required(VERSION 3.16)

project(K-means_II)

set(CMAKE_CXX_STANDARD 14)
find_package(OpenMP REQUIRED)


enable_language(CUDA)
enable_testing()

add_subdirectory(./utils)
add_subdirectory(./k-means_II)
add_subdirectory(./PQ)