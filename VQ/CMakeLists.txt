include_directories(./include)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_library(VQ SHARED ./src/vq.cu)
set_target_properties(VQ PROPERTIES CUDA_ARCHITECTURES "50;60;70;80")
target_link_libraries(VQ common_utils_cuda)
target_link_libraries(VQ random_k-means)
target_link_libraries(VQ OpenMP::OpenMP_CXX)

add_executable(test_vq ./test/test_vq.cu)
target_link_libraries(test_vq VQ)
set_target_properties(test_vq PROPERTIES CUDA_ARCHITECTURES "50;60;70;80")
add_test(NAME test_vq COMMAND test_vq)