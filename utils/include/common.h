#pragma once

#include <stddef.h>
#include <iostream>
#include <string>
#include <fstream>
#include "../include/config.h"

float euclideanDistance(float *x, float *y, const int dim);                                                                                                     // 欧式距离
float costFromV2S(float *x, float *cluster_set, const int dim, const unsigned int size);                                                                        // 向量与集合之间的代价
float costFromS2S(float *original_set, float *cluster_set, const int dim, const unsigned int original_size, const unsigned int cluster_size);                   // 集合与集合之间的代价
unsigned int belongV2S(float *x, float *cluster_set, const int dim, const unsigned int size);                                                                   // 判断向量与聚类中心集中向量的归属关系
void belongS2S(unsigned int *index, float *original_set, float *cluster_set, const int dim, const unsigned int original_size, const unsigned int cluster_size); // 判断向量集中向量与聚类中心集中向量的归属关系

void kmeanspp(float *cluster_final, float *cluster_set, unsigned int *omega, unsigned int k, const int dim, const unsigned int cluster_size); // k-means++

void meanVec(float *res, float *original_set, unsigned int *belong, const int dim, const unsigned int original_size, const unsigned int index); // 计算集合中向量的均值向量
bool isClose(float *cluster_new, float *cluster_old, const int dim, const unsigned int cluster_size, float epsilon);                            // 判断两个聚类中心集是否接近

void save(float *data, unsigned int size, const std::string &filename);
void load(float *data, unsigned int size, const std::string &filename);
void save(unsigned int *data, unsigned int size, const std::string &filename);
void load(unsigned int *data, unsigned int size, const std::string &filename);

void split_file(float *original_data, unsigned int size, int dim, unsigned int m, std::string prefix);