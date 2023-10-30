#pragma once

class kmeans
{
public:
    kmeans(float *original_data, size_t original_size, size_t k)
        : original_data(original_data), original_size(original_size), k(k) {}

    void init();      // 初始化过程
    void iteration(); // 迭代过程

private:
    float *original_data; // 原始向量集
    size_t original_size; // 原始向量个数
    float *center;        // 聚类中心集
    size_t k;             // 聚类中心个数

    // 超参数
    float l;       // 超采样因子
    float epsilon; // 迭代终止条件阈值
};