#pragma once

#include <stddef.h>

float euclideanDistance(float *x, float *y, const int dim);                                                                 // 欧式距离
float costFromV2S(float *x, float *cluster_set, const int dim, const size_t size);                                             // 向量与集合之间的代价
float costFromS2S(float *original_set, float *cluster_set, const int dim, const size_t original_size, const size_t cluster_size); // 集合与集合之间的代价
size_t belongV2S(float *x, float *cluster_set, const int dim, const size_t size);                                              // 判断向量与聚类中心集中向量的归属关系
size_t *belongS2S(float *original_set, float *cluster_set, const int dim, const size_t original_size, const size_t cluster_size); // 判断向量集中向量与聚类中心集中向量的归属关系

float *kmeanspp(float *cluster_set, size_t *omega, size_t k, const int dim, const size_t cluster_size); // k-means++