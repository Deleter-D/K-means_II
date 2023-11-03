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

    size_t vec_bytes = original_dim * sizeof(float);
    size_t cluster_bytes = K * vec_bytes;

    size_t index = distrib(gen);

    memset(cluster_set, 0, (OVER_SAMPLING * INIT_ITERATION_TIMES + 1) * vec_bytes);

    // 记录聚类中心向量的全局索引
    std::set<size_t> center_index;

    // 将第一个聚类中心并入集合C
    memcpy(cluster_set, &original_data[index * original_dim], vec_bytes);
    current_k = 1;
    center_index.insert(index); // 记录第一个聚类中心全局索引

    // 计算此时聚类中心集与全集的代价
    float phi;
#ifdef __USE_CUDA__
    phi = cudaCostFromS2S(original_data, cluster_set, original_dim, original_size, current_k);
#else
    phi = costFromS2S(original_data, cluster_set, original_dim, original_size, current_k);
#endif

    // 存放概率值和索引的key-value对
    std::multimap<float, size_t, std::greater<float>> probability;

    // 迭代
    for (int i = 0; i < INIT_ITERATION_TIMES; i++)
    {

        float cost_set2cluster;
#ifdef __USE_CUDA__
        cost_set2cluster = cudaCostFromS2S(original_data, cluster_set, original_dim, original_size, current_k);
#else
        cost_set2cluster = costFromS2S(original_data, cluster_set, original_dim, original_size, current_k);
#endif

        // 计算所有向量的概率
        float *current_ps = (float *)malloc(original_size * sizeof(float));
        for (int j = 0; j < original_size; j++)
        {
            float *temp = &original_data[j * original_dim];
#ifdef __USE_CUDA__
            current_ps[j] = OVER_SAMPLING * cudaCostFromV2S(temp, cluster_set, original_dim, current_k) / cost_set2cluster;
#else
            current_ps[j] = OVER_SAMPLING * costFromV2S(temp, cluster_set, original_dim, current_k) / cost_set2cluster;
#endif
        }
        // 按从大到小的顺序记录"概率-索引"对
        for (int j = 0; j < original_size; j++)
        {
            probability.insert({current_ps[j], j});
        }
        free(current_ps);

        auto beg = probability.cbegin();
        auto end = probability.cend();
        // 将map中的前l个向量并入聚类中心集
        for (int i = 0; i < OVER_SAMPLING; i++)
        {
            auto temp_index = beg->second;
            if (beg != end && !center_index.count(temp_index))
            {
                memcpy(&cluster_set[original_dim * current_k], &original_data[temp_index * original_dim], vec_bytes);
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

    size_t *index_X2C = (size_t *)malloc(original_size * sizeof(size_t));
#ifdef __USE_CUDA__
    cudaBelongS2S(index_X2C, original_data, cluster_set, original_dim, original_size, current_k);
#else
    belongS2S(index_X2C, original_data, cluster_set, original_dim, original_size, current_k);
#endif

    for (int i = 0; i < original_size; i++)
    {
        omega[index_X2C[i]]++;
    }

    free(index_X2C);

    // 利用kmeans++获取最终聚类中心
    float *cluster_final = (float *)malloc(cluster_bytes);
#ifdef __USE_CUDA__
    cudaKmeanspp(cluster_final, cluster_set, omega, K, original_dim, current_k);
#else
    kmeanspp(cluster_final, cluster_set, omega, K, original_dim, current_k);
#endif
    free(omega);

    cluster_set = (float *)realloc(cluster_set, cluster_bytes);
    memcpy(cluster_set, cluster_final, cluster_bytes);
    free(cluster_final);
}

void kmeans_II::iteration()
{
    size_t vec_bytes = original_dim * sizeof(float);
    size_t cluster_bytes = K * vec_bytes;

    // 计算全集中的向量所属的聚类中心的索引
    size_t *belong = (size_t *)malloc(original_size * sizeof(size_t));
#ifdef __USE_CUDA__
    cudaBelongS2S(belong, original_data, cluster_set, original_dim, original_size, K);
#else
    belongS2S(belong, original_data, cluster_set, original_dim, original_size, K);
#endif

    float *cluster_new = (float *)malloc(cluster_bytes);
    memcpy(cluster_new, cluster_set, cluster_bytes);

    int iteration_times = 0;
    bool isclose;

    size_t *temp_count = (size_t *)malloc(K * sizeof(size_t)); // 删除
    do
    {
        memcpy(cluster_set, cluster_new, cluster_bytes);

#ifdef __USE_CUDA__
        cudaGetNewCluster(cluster_new, original_data, belong, original_dim, original_size);
#else
        // 产生新的聚类中心集
        float *mean_vec = (float *)malloc(vec_bytes);
        memset(mean_vec, 0, vec_bytes);
        for (int i = 0; i < K; i++)
        {
            memset(mean_vec, 0, vec_bytes);
            meanVec(mean_vec, original_data, belong, original_dim, original_size, i);
            memcpy(&cluster_new[i * original_dim], mean_vec, vec_bytes);
        }
        free(mean_vec);
#endif

#ifdef __USE_CUDA__
        cudaBelongS2S(belong, original_data, cluster_new, original_dim, original_size, K);
#else
        // 更新归属关系索引
        belongS2S(belong, original_data, cluster_new, original_dim, original_size, K);
#endif

        // 删除
        printf("迭代%d\n", iteration_times);
        memset(temp_count, 0, K * sizeof(size_t));
        for (int i = 0; i < original_size; i++)
        {
            temp_count[belong[i]]++;
        }
        for (int i = 0; i < K; i++)
        {
            printf("%d: %ld个\n", i, temp_count[i]);
        }

        iteration_times++;

#ifdef __USE_CUDA__
        isclose = cudaIsClose(cluster_new, cluster_set, original_dim, K, THRESHOLD);
#else
        isclose = isClose(cluster_new, cluster_set, original_dim, K, THRESHOLD);
#endif
        printf("迭代结果%d: %s\n", iteration_times, isclose ? "close" : "not close"); // 删除
    } while (!isclose && iteration_times < MAX_KMEANS_ITERATION_TIMES);

    free(belong);
}