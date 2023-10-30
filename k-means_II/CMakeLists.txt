include_directories(./include)

add_library(k-means_II SHARED ./src/kmeans.cpp)
install(TARGETS k-means_II DESTINATION ${PROJECT_SOURCE_DIR}/install)
