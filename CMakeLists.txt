cmake_minimum_required(VERSION 3.16)

project(K-means_II)

set(CMAKE_CXX_STANDARD 14)
find_package(OpenMP REQUIRED)


enable_language(CUDA)
enable_testing()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g")

add_subdirectory(./utils)
add_subdirectory(./k-means_II)
add_subdirectory(./PQ)

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
aux_source_directory(./utils/src SOURCES)
aux_source_directory(./k-means_II/src SOURCES)
aux_source_directory(./PQ/src SOURCES)
add_library(kmeansPQII SHARED ${SOURCES})
target_link_libraries(kmeansPQII PRIVATE common_utils common_utils_cuda k-means_II mini_batch_k-means PQ)
set_target_properties(kmeansPQII PROPERTIES CUDA_ARCHITECTURES "50;60;70;80")
install(TARGETS kmeansPQII DESTINATION lib)