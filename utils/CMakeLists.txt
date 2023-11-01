include_directories(./include)

add_library(common_utils SHARED ./src/common.cpp)
install(TARGETS common_utils DESTINATION ${PROJECT_SOURCE_DIR}/install)

add_library(common_utils_cuda SHARED ./src/common.cu)
set_target_properties(common_utils_cuda PROPERTIES CUDA_ARCHITECTURES "50;60;70;80")
install(TARGETS common_utils_cuda DESTINATION ${PROJECT_SOURCE_DIR}/install)

add_executable(test_euclidean ./test/cpu/test_euclidean.cpp)
target_link_libraries(test_euclidean common_utils)
add_test(NAME test_euclidean COMMAND test_euclidean)

add_executable(test_cost_V2S ./test/cpu/test_cost_V2S.cpp)
target_link_libraries(test_cost_V2S common_utils)
add_test(NAME test_cost_V2S COMMAND test_cost_V2S)

add_executable(test_cost_S2S ./test/cpu/test_cost_S2S.cpp)
target_link_libraries(test_cost_S2S common_utils)
add_test(NAME test_cost_S2S COMMAND test_cost_S2S)

add_executable(test_belong_V2S ./test/cpu/test_belong_V2S.cpp)
target_link_libraries(test_belong_V2S common_utils)
add_test(NAME test_belong_V2S COMMAND test_belong_V2S)

add_executable(test_belong_S2S ./test/cpu/test_belong_S2S.cpp)
target_link_libraries(test_belong_S2S common_utils)
add_test(NAME test_belong_S2S COMMAND test_belong_S2S)

add_executable(test_kmeanspp ./test/cpu/test_kmeanspp.cpp)
target_link_libraries(test_kmeanspp common_utils)
add_test(NAME test_kmeanspp COMMAND test_kmeanspp)

add_executable(test_meanOfV ./test/cpu/test_meanOfV.cpp)
target_link_libraries(test_meanOfV common_utils)
add_test(NAME test_meanOfV COMMAND test_meanOfV)

add_executable(test_isClose ./test/cpu/test_isClose.cpp)
target_link_libraries(test_isClose common_utils)
add_test(NAME test_isClose COMMAND test_isClose)

add_executable(test_cuda_euclidean ./test/cuda/test_cuda_euclidean.cu)
target_link_libraries(test_cuda_euclidean common_utils common_utils_cuda)
set_target_properties(test_cuda_euclidean PROPERTIES CUDA_ARCHITECTURES "50;60;70;80")
add_test(NAME test_cuda_euclidean COMMAND test_cuda_euclidean)

add_executable(test_cuda_cost_V2S ./test/cuda/test_cuda_cost_V2S.cu)
target_link_libraries(test_cuda_cost_V2S common_utils common_utils_cuda)
set_target_properties(test_cuda_cost_V2S PROPERTIES CUDA_ARCHITECTURES "50;60;70;80")
add_test(NAME test_cuda_cost_V2S COMMAND test_cuda_cost_V2S)

add_executable(test_cuda_cost_S2S ./test/cuda/test_cuda_cost_S2S.cu)
target_link_libraries(test_cuda_cost_S2S common_utils common_utils_cuda)
set_target_properties(test_cuda_cost_S2S PROPERTIES CUDA_ARCHITECTURES "50;60;70;80")
add_test(NAME test_cuda_cost_S2S COMMAND test_cuda_cost_S2S)

add_executable(test_cuda_belong_V2S ./test/cuda/test_cuda_belong_V2S.cu)
target_link_libraries(test_cuda_belong_V2S common_utils common_utils_cuda)
set_target_properties(test_cuda_belong_V2S PROPERTIES CUDA_ARCHITECTURES "50;60;70;80")
add_test(NAME test_cuda_belong_V2S COMMAND test_cuda_belong_V2S)

add_executable(test_cuda_belong_S2S ./test/cuda/test_cuda_belong_S2S.cu)
target_link_libraries(test_cuda_belong_S2S common_utils common_utils_cuda)
set_target_properties(test_cuda_belong_S2S PROPERTIES CUDA_ARCHITECTURES "50;60;70;80")
add_test(NAME test_cuda_belong_S2S COMMAND test_cuda_belong_S2S)

add_executable(test_cuda_kmeanspp ./test/cuda/test_cuda_kmeanspp.cu)
target_link_libraries(test_cuda_kmeanspp common_utils common_utils_cuda)
set_target_properties(test_cuda_kmeanspp PROPERTIES CUDA_ARCHITECTURES "50;60;70;80")
add_test(NAME test_cuda_kmeanspp COMMAND test_cuda_kmeanspp)

add_executable(test_cuda_new_cluster ./test/cuda/test_cuda_new_cluster.cu)
target_link_libraries(test_cuda_new_cluster common_utils common_utils_cuda)
set_target_properties(test_cuda_new_cluster PROPERTIES CUDA_ARCHITECTURES "50;60;70;80")
add_test(NAME test_cuda_new_cluster COMMAND test_cuda_new_cluster)