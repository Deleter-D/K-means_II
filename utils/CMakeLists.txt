include_directories(./include)

add_library(common_utils SHARED ./src/common.cpp)
install(TARGETS common_utils DESTINATION ${PROJECT_SOURCE_DIR}/install)

add_executable(test_euclidean ./test/test_euclidean.cpp)
target_link_libraries(test_euclidean common_utils)
add_test(NAME test_euclidean COMMAND test_euclidean)

add_executable(test_cost_V2S ./test/test_cost_V2S.cpp)
target_link_libraries(test_cost_V2S common_utils)
add_test(NAME test_cost_V2S COMMAND test_cost_V2S)

add_executable(test_cost_S2S ./test/test_cost_S2S.cpp)
target_link_libraries(test_cost_S2S common_utils)
add_test(NAME test_cost_S2S COMMAND test_cost_S2S)

add_executable(test_belong_V2S ./test/test_belong_V2S.cpp)
target_link_libraries(test_belong_V2S common_utils)
add_test(NAME test_belong_V2S COMMAND test_belong_V2S)

add_executable(test_belong_S2S ./test/test_belong_S2S.cpp)
target_link_libraries(test_belong_S2S common_utils)
add_test(NAME test_belong_S2S COMMAND test_belong_S2S)

add_executable(test_kmeanspp ./test/test_kmeanspp.cpp)
target_link_libraries(test_kmeanspp common_utils)
add_test(NAME test_kmeanspp COMMAND test_kmeanspp)

add_executable(test_meanOfV ./test/test_meanOfV.cpp)
target_link_libraries(test_meanOfV common_utils)
add_test(NAME test_meanOfV COMMAND test_meanOfV)

add_executable(test_isClose ./test/test_isClose.cpp)
target_link_libraries(test_isClose common_utils)
add_test(NAME test_isClose COMMAND test_isClose)