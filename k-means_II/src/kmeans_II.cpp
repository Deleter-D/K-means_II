#include <math.h>
#include <cstring>
#include <random>
#include <functional>
#include <map>
#include <set>
#include <iostream>
#include <cmath>

#include "../include/kmeans_II.h"
#include "../../utils/include/common.h"
#include "../../utils/include/common.cuh"

#define __USE_CUDA__

void kmeans_II::init()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, original_size - 1);

    size_t index = distrib(gen);

    // 分配2 * k * ITERATION_TIMES个聚类中心的内存
    size_t center_bytes = (OVER_SAMPLING * INIT_ITERATION_TIMES + 1) * original_dim * sizeof(float);
    cluster_set = (float *)malloc(center_bytes);
    memset(cluster_set, 0, center_bytes);

    // 记录聚类中心向量的全局索引
    std::set<size_t> center_index;

    // 将第一个聚类中心并入集合C
    memcpy(cluster_set, &original_data[index * original_dim], original_dim * sizeof(float));
    current_k = 1;
    center_index.insert(index); // 记录第一个聚类中心全局索引

    // 计算此时聚类中心集与全集的代价
    float phi;
#ifdef __USE_CUDA__
    printf("cudaCostFromS2S\n");
    phi = cudaCostFromS2S(original_data, cluster_set, original_dim, original_size, current_k);
#else
    printf("costFromS2S\n");
    phi = costFromS2S(original_data, cluster_set, original_dim, original_size, current_k);
#endif

    // 迭代
    for (int i = 0; i < INIT_ITERATION_TIMES; i++)
    {
        // 存放概率值和索引的key-value对
        std::multimap<float, size_t, std::greater<float>> probability;
        float current_p;
        float cost_set2cluster;
#ifdef __USE_CUDA__
        cost_set2cluster = cudaCostFromS2S(original_data, cluster_set, original_dim, original_size, current_k);
#else
        cost_set2cluster = costFromS2S(original_data, cluster_set, original_dim, original_size, current_k);
#endif
        for (int j = 0; j < original_size; j++)
        {
            float *temp = &original_data[j * original_dim];
// 计算当前向量的概率
#ifdef __USE_CUDA__
            current_p = OVER_SAMPLING * cudaCostFromV2S(temp, cluster_set, original_dim, current_k) / cost_set2cluster;
#else
            current_p = OVER_SAMPLING * costFromV2S(temp, cluster_set, original_dim, current_k) / cost_set2cluster;
#endif
            probability.insert({current_p, j});
        }
        auto beg = probability.cbegin();
        auto end = probability.cend();
        // 将map中的前l个向量并入聚类中心集
        for (int i = 0; i < OVER_SAMPLING; i++)
        {
            auto temp_index = beg->second;
            if (beg != end && !center_index.count(temp_index))
            {
                memcpy(&cluster_set[original_dim * current_k], &original_data[temp_index * original_dim], original_dim * sizeof(float));
                current_k++;                     // 聚类中心当前数量自增
                center_index.insert(temp_index); // 记录并入聚类中心集的全局索引
            }
            beg++;
        }
        probability.clear();
    }

    // 记录每个聚类中心的权重
    size_t *omega = (size_t *)malloc(current_k * sizeof(size_t));
    memset(omega, 0, current_k * sizeof(size_t));
    // 记录每个向量归属的聚类中心索引

    size_t *index_X2C;
#ifdef __USE_CUDA__
    index_X2C = cudaBelongS2S(original_data, cluster_set, original_dim, original_size, current_k);
#else
    index_X2C = belongS2S(original_data, cluster_set, original_dim, original_size, current_k);
#endif

    for (int i = 0; i < original_size; i++)
    {
        omega[index_X2C[i]]++;
    }

    // 利用kmeans++获取最终聚类中心

    float *cluster_final;
#ifdef __USE_CUDA__
    cluster_final = cudaKmeanspp(cluster_set, omega, K, original_dim, current_k);
#else
    cluster_final = kmeanspp(cluster_set, omega, K, original_dim, current_k);
#endif

    cluster_set = cluster_final;
}

void kmeans_II::iteration()
{
    // 计算全集中的向量所属的聚类中心的索引
    size_t *belong;
#ifdef __USE_CUDA__
    belong = cudaBelongS2S(original_data, cluster_set, original_dim, original_size, K);
#else
    belong = belongS2S(original_data, cluster_set, original_dim, original_size, K);
#endif

    float *cluster_new = (float *)malloc(K * original_dim * sizeof(float));
    memcpy(cluster_new, cluster_set, K * original_dim * sizeof(float));

    int iteration_times = 0;
    bool isclose;

    size_t *temp_count = (size_t *)malloc(K * sizeof(size_t));
    do
    {
        memcpy(cluster_set, cluster_new, K * original_dim * sizeof(float));

#ifdef __USE_CUDA__
        cudaGetNewCluster(cluster_new, original_data, belong, original_dim, original_size);
#else
        // 产生新的聚类中心集
        for (int i = 0; i < K; i++)
        {
            memcpy(&cluster_new[i * original_dim], meanVec(original_data, belong, original_dim, original_size, i), original_dim * sizeof(float));
        }
#endif

#ifdef __USE_CUDA__
        belong = cudaBelongS2S(original_data, cluster_new, original_dim, original_size, K);
#else
        // 更新归属关系索引
        belong = belongS2S(original_data, cluster_new, original_dim, original_size, K);
#endif

        printf("迭代%d\n", iteration_times);
        memset(temp_count, 0, K * sizeof(size_t));
        // for (int i = 0; i < original_size; i++)
        // {
        //     printf("%d,%d\n", belong[i], belong_cuda[i]);
        // }
        for (int i = 0; i < original_size; i++)
        {
            temp_count[belong[i]]++;
        }
        for (int i = 0; i < K; i++)
        {
            printf("%d: %d个\n", i, temp_count[i]);
        }

        iteration_times++;

#ifdef __USE_CUDA__
        isclose = cudaIsClose(cluster_new, cluster_set, original_dim, K, THRESHOLD);
#else
        isclose = isClose(cluster_new, cluster_set, original_dim, K, THRESHOLD);
#endif
        printf("迭代结果%d: %s\n", iteration_times, isclose ? "close" : "not close");
    } while (!isclose && iteration_times < MAX_KMEANS_ITERATION_TIMES);
    printf("------------->%d\n", iteration_times);
}