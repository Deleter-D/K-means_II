#pragma once

#define ITERATION_TIMES 5

class kmeans
{
public:
    kmeans() {}
    kmeans(float *original_data, size_t original_size, size_t original_dim, size_t k)
        : original_data(original_data),
          original_size(original_size),
          original_dim(original_dim),
          k(k) {}

    void init();      // 初始化过程
    void iteration(); // 迭代过程

public:
    float euclideanDistance(float *x, float *y, const int dim);                                   // 欧式距离
    float costFromV2S(float *x, float *set, const int dim, const int size);                       // 向量与集合之间的代价
    float costFromS2S(float *set1, float *set2, const int dim, const int size1, const int size2); // 集合与集合之间的代价

public:
    float *original_data; // 原始向量集
    size_t original_size; // 原始向量个数
    size_t original_dim;  // 原始向量维度
    float *center;        // 聚类中心集
    size_t current_k;     // 聚类中心的当前个数
    size_t k;             // 聚类中心的目标个数

    // 超参数
    float l;       // 超采样因子
    float epsilon; // 迭代终止条件阈值
};