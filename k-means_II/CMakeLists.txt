include_directories(./include)

add_library(k-means_II SHARED ./src/kmeans_II.cpp)
target_link_libraries(k-means_II common_utils)
install(TARGETS k-means_II DESTINATION ${PROJECT_SOURCE_DIR}/install)

add_executable(test_init ./test/test_init.cpp)
target_link_libraries(test_init k-means_II)
add_test(NAME test_init COMMAND test_init)