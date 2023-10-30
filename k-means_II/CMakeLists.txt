include_directories(./include)

add_library(k-means_II SHARED ./src/kmeans.cpp)
install(TARGETS k-means_II DESTINATION ${PROJECT_SOURCE_DIR}/install)

add_executable(test_euclidean ./test/test_euclidean.cpp)
target_link_libraries(test_euclidean k-means_II)
add_test(NAME test_euclidean COMMAND test_euclidean)

add_executable(test_cost_V2S ./test/test_cost_V2S.cpp)
target_link_libraries(test_cost_V2S k-means_II)
add_test(NAME test_cost_V2S COMMAND test_cost_V2S)

add_executable(test_cost_S2S ./test/test_cost_S2S.cpp)
target_link_libraries(test_cost_S2S k-means_II)
add_test(NAME test_cost_S2S COMMAND test_cost_S2S)

add_executable(test_init ./test/test_init.cpp)
target_link_libraries(test_init k-means_II)
add_test(NAME test_init COMMAND test_init)