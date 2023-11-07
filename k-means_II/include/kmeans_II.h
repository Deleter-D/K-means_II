#pragma once

#include <stddef.h>
#include "../../utils/include/config.h"

void init(float *original_data, size_t original_size, size_t original_dim, float *cluster_set);      // 初始化过程
void iteration(float *original_data, size_t original_size, size_t original_dim, float *cluster_set); // 迭代过程
void kmeansII(float *original_data, size_t original_size, size_t original_dim, float *cluster_set);

// class kmeans_II
// {
// public:
//     kmeans_II()
//     {
//         // 分配OVER_SAMPLING * ITERATION_TIMES + 1个聚类中心的内存
//         cluster_set = (float *)malloc((OVER_SAMPLING * INIT_ITERATION_TIMES + 1) * original_dim * sizeof(float));
//     }
//     kmeans_II(float *original_data, size_t original_size, size_t original_dim)
//         : original_data(original_data),
//           original_size(original_size),
//           original_dim(original_dim)
//     {
//         cluster_set = (float *)malloc((OVER_SAMPLING * INIT_ITERATION_TIMES + 1) * original_dim * sizeof(float));
//     }

//     ~kmeans_II()
//     {
//         free(cluster_set);
//     }

//     void init(float *original_data, size_t original_size, size_t original_dim, float *cluster_set);      // 初始化过程
//     void iteration(float *original_data, size_t original_size, size_t original_dim, float *cluster_set); // 迭代过程

// public:
//     // float *original_data; // 原始向量集
//     // size_t original_size; // 原始向量个数
//     // size_t original_dim;  // 原始向量维度
//     // float *cluster_set;   // 聚类中心集
//     // size_t current_k;     // 聚类中心的当前个数

//     // 超参数
//     float l;       // 超采样因子
//     float epsilon; // 迭代终止条件阈值
// };