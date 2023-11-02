include_directories(./include)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_library(k-means_II SHARED ./src/kmeans_II.cpp)
target_link_libraries(k-means_II common_utils)
target_link_libraries(k-means_II common_utils_cuda)
install(TARGETS k-means_II DESTINATION ${PROJECT_SOURCE_DIR}/install)

# add_executable(test_init ./test/test_init.cpp)
# target_link_libraries(test_init k-means_II)
# add_test(NAME test_init COMMAND test_init)

add_executable(test_kmeans_II ./test/test_kmeans_II.cpp)
target_link_libraries(test_kmeans_II k-means_II)
add_test(NAME test_kmeans_II COMMAND test_kmeans_II)