cmake_minimum_required(VERSION 3.16)

project(ANN)

set(CMAKE_CXX_STANDARD 14)
find_package(OpenMP REQUIRED)


enable_language(CUDA)
enable_testing()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g")

add_subdirectory(./utils)
add_subdirectory(./k-means)
add_subdirectory(./PQ)
add_subdirectory(./VQ)

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
aux_source_directory(./utils/src SOURCES)
aux_source_directory(./k-means/src SOURCES)
aux_source_directory(./PQ/src SOURCES)
aux_source_directory(./VQ/src SOURCES)
add_library(ANNSearch SHARED ${SOURCES})
target_link_libraries(ANNSearch PRIVATE common_utils common_utils_cuda k-means_II random_k-means PQ VQ)
set_target_properties(ANNSearch PROPERTIES CUDA_ARCHITECTURES "50;60;70;80")
install(TARGETS ANNSearch DESTINATION lib)