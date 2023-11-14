#pragma once

#include <stddef.h>
#include <iostream>
#include <string>
#include <fstream>
#include "../include/config.h"

float euclideanDistance(float *x, float *y, const int dim);                                                                                   // 欧式距离
float costFromV2S(float *x, float *cluster_set, const int dim, const size_t size);                                                            // 向量与集合之间的代价
float costFromS2S(float *original_set, float *cluster_set, const int dim, const size_t original_size, const size_t cluster_size);             // 集合与集合之间的代价
size_t belongV2S(float *x, float *cluster_set, const int dim, const size_t size);                                                             // 判断向量与聚类中心集中向量的归属关系
void belongS2S(size_t *index, float *original_set, float *cluster_set, const int dim, const size_t original_size, const size_t cluster_size); // 判断向量集中向量与聚类中心集中向量的归属关系

void kmeanspp(float *cluster_final, float *cluster_set, size_t *omega, size_t k, const int dim, const size_t cluster_size); // k-means++

void meanVec(float *res, float *original_set, size_t *belong, const int dim, const size_t original_size, const size_t index); // 计算集合中向量的均值向量
bool isClose(float *cluster_new, float *cluster_old, const int dim, const size_t cluster_size, float epsilon);                // 判断两个聚类中心集是否接近

void save(float *data, size_t size, const std::string &filename);
void load(float *data, size_t size, const std::string &filename);
void save(size_t *data, size_t size, const std::string &filename);
void load(size_t *data, size_t size, const std::string &filename);

void split_file(float *original_data, size_t size, int dim, unsigned int m, std::string prefix);