include_directories(./include)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_library(k-means_II SHARED ./src/kmeans_II.cpp)
target_link_libraries(k-means_II common_utils)
target_link_libraries(k-means_II common_utils_cuda)
target_link_libraries(k-means_II OpenMP::OpenMP_CXX)

add_library(mini_batch_k-means SHARED ./src/mini_batch_kmeans.cpp)
target_link_libraries(mini_batch_k-means common_utils)
target_link_libraries(mini_batch_k-means common_utils_cuda)
target_link_libraries(mini_batch_k-means OpenMP::OpenMP_CXX)

add_executable(test_init ./test/test_init.cpp)
target_link_libraries(test_init k-means_II)
add_test(NAME test_init COMMAND test_init)

add_executable(test_kmeans_II ./test/test_kmeans_II.cpp)
target_link_libraries(test_kmeans_II k-means_II)
add_test(NAME test_kmeans_II COMMAND test_kmeans_II)