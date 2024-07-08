#include <math.h>
#include <cstring>
#include <random>
#include <functional>
#include <map>
#include <set>
#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <omp.h>
#include <time.h>

#include "../include/kmeans_II.h"
#include "../../utils/include/common.h"
#include "../../utils/include/common.cuh"

#define __USE_CUDA__

void init(float *original_data, unsigned int original_size, unsigned int original_dim, float *cluster_set)
{
#ifdef DEBUG
#ifdef _OPENMP
    std::cout << DEBUG_HEAD << "openmp is enabled.\n";
#else
    std::cout << DEBUG_HEAD << "openmp is not enabled.\n";
#endif
#ifdef __USE_CUDA__
    std::cout << DEBUG_HEAD << "cuda is enabled.\n";
#else
    std::cout << DEBUG_HEAD << "cuda is not enabled.\n";
#endif
#endif
    time_t start_time = 0;
    time_t end_time = 0;
    start_time = time(NULL);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, original_size - 1);

    size_t vec_bytes = original_dim * sizeof(float);
    size_t cluster_bytes = K * vec_bytes;

    unsigned int index = distrib(gen);

    float *cluster_set_temp = (float *)malloc((OVER_SAMPLING * INIT_ITERATION_TIMES + 1) * vec_bytes);
    memset(cluster_set_temp, 0, (OVER_SAMPLING * INIT_ITERATION_TIMES + 1) * vec_bytes);

    // 记录聚类中心向量的全局索引
    std::set<unsigned int> center_index;

    // 将第一个聚类中心并入集合C
    memcpy(cluster_set_temp, &original_data[index * original_dim], vec_bytes);
    unsigned int current_k = 1;
    center_index.insert(index); // 记录第一个聚类中心全局索引

#ifdef DEBUG
    std::cout << DEBUG_HEAD << "begin to compute phi.\n";
#endif
    // 计算此时聚类中心集与全集的代价
    float phi;
#ifdef __USE_CUDA__
    phi = cudaCostFromS2S(original_data, cluster_set_temp, original_dim, original_size, current_k);
#else
    phi = costFromS2S(original_data, cluster_set_temp, original_dim, original_size, current_k);
#endif

#ifdef DEBUG
    std::cout << DEBUG_HEAD << "Compute phi finished.\n";
#endif

    // 存放概率值和索引的key-value对
    std::vector<std::pair<float, unsigned int>> probability(original_size);
    // 迭代
    for (int i = 0; i < INIT_ITERATION_TIMES; i++)
    {
#ifdef DEBUG
        std::cout << DEBUG_HEAD << "The " << i << "th initial iteration begining.\n";
#endif
        float cost_set2cluster;
#ifdef __USE_CUDA__
        cost_set2cluster = cudaCostFromS2S(original_data, cluster_set_temp, original_dim, original_size, current_k);
#else
        cost_set2cluster = costFromS2S(original_data, cluster_set_temp, original_dim, original_size, current_k);
#endif

        // 计算所有向量的概率
        float *current_ps = (float *)malloc(original_size * sizeof(float));
        for (int j = 0; j < original_size; j++)
        {
            float *temp = &original_data[j * original_dim];
#ifdef __USE_CUDA__
            current_ps[j] = OVER_SAMPLING * cudaCostFromV2S(temp, cluster_set_temp, original_dim, current_k) / cost_set2cluster;
#else
            current_ps[j] = OVER_SAMPLING * costFromV2S(temp, cluster_set_temp, original_dim, current_k) / cost_set2cluster;
#endif
        }

#pragma omp parallel for
        for (int j = 0; j < original_size; j++)
        {
            probability[j] = std::make_pair(current_ps[j], j);
        }
        free(current_ps);

        // 排序，只排序l个
        std::partial_sort(probability.begin(), probability.begin() + OVER_SAMPLING, probability.end(), std::greater<std::pair<float, unsigned int>>());

        auto beg = probability.cbegin();
        auto end = probability.cend();
        // 将map中的前l个向量并入聚类中心集
        for (int i = 0; i < OVER_SAMPLING; i++)
        {
            auto temp_index = beg->second;
            if (beg != end && !center_index.count(temp_index))
            {
                memcpy(&cluster_set_temp[original_dim * current_k], &original_data[temp_index * original_dim], vec_bytes);
                current_k++;                     // 聚类中心当前数量自增
                center_index.insert(temp_index); // 记录并入聚类中心集的全局索引
            }
            beg++;
        }
#ifdef DEBUG
        std::cout << DEBUG_HEAD << "The " << i << "th initial iteration finished.\n";
#endif
    }

    // 记录每个聚类中心的权重
    unsigned int *omega = (unsigned int *)malloc(current_k * sizeof(unsigned int));
    memset(omega, 0, current_k * sizeof(unsigned int));

    // 记录每个向量归属的聚类中心索引
    unsigned int *index_X2C = (unsigned int *)malloc(original_size * sizeof(unsigned int));
#ifdef __USE_CUDA__
    cudaBelongS2S(index_X2C, original_data, cluster_set_temp, original_dim, original_size, current_k);
#else
    belongS2S(index_X2C, original_data, cluster_set_temp, original_dim, original_size, current_k);
#endif
#pragma omp parallel for
    for (int i = 0; i < original_size; i++)
    {
        unsigned int index = index_X2C[i];
#pragma omp atomic
        omega[index]++;
    }

    free(index_X2C);

#ifdef DEBUG
    printf("%sCompute omega finished.\n", DEBUG_HEAD);
#endif

    // 利用kmeans++获取最终聚类中心
    float *cluster_final = (float *)malloc(cluster_bytes);
#ifdef __USE_CUDA__
    cudaKmeanspp(cluster_final, cluster_set_temp, omega, K, original_dim, current_k);
#else
    kmeanspp(cluster_final, cluster_set_temp, omega, K, original_dim, current_k);
#endif
    free(omega);
    free(cluster_set_temp);

#ifdef DEBUG
    printf("%sK-means++ finished.\n", DEBUG_HEAD);
#endif

    memcpy(cluster_set, cluster_final, cluster_bytes);
    free(cluster_final);

#ifdef DEBUG
    printf("%sintilization finished.\n", DEBUG_HEAD);
#endif

    end_time = time(NULL);
#ifdef DEBUG
    printf("耗时：%lds\n", end_time - start_time);
#endif
}

void iteration(float *original_data, unsigned int original_size, unsigned int original_dim, float *cluster_set)
{
    size_t vec_bytes = original_dim * sizeof(float);
    size_t cluster_bytes = K * vec_bytes;

    // 计算全集中的向量所属的聚类中心的索引
    unsigned int *belong = (unsigned int *)malloc(original_size * sizeof(unsigned int));
#ifdef __USE_CUDA__
    cudaBelongS2S(belong, original_data, cluster_set, original_dim, original_size, K);
#else
    belongS2S(belong, original_data, cluster_set, original_dim, original_size, K);
#endif

    float *cluster_new = (float *)malloc(cluster_bytes);
    memcpy(cluster_new, cluster_set, cluster_bytes);

    int iteration_times = 0;
    bool isclose;

    unsigned int *temp_count = (unsigned int *)malloc(K * sizeof(unsigned int)); // 删除
    do
    {
        memcpy(cluster_set, cluster_new, cluster_bytes);

#ifdef __USE_CUDA__
        cudaGetNewCluster(cluster_new, original_data, belong, original_dim, original_size, K);
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

#ifdef DEBUG
        printf("迭代%d\n", iteration_times);
        memset(temp_count, 0, K * sizeof(unsigned int));

        for (int i = 0; i < original_size; i++)
        {
            unsigned int index = belong[i];

            temp_count[index]++;
        }

        for (int i = 0; i < K; i++)
        {
            printf("%d: %d个\n", i, temp_count[i]);
        }
#endif

        iteration_times++;

#ifdef __USE_CUDA__
        isclose = cudaIsClose(cluster_new, cluster_set, original_dim, K, THRESHOLD);
#else
        isclose = isClose(cluster_new, cluster_set, original_dim, K, THRESHOLD);
#endif

#ifdef DEBUG
        printf("迭代结果%d: %s\n", iteration_times, isclose ? "close" : "not close");
#endif

    } while (!isclose && iteration_times < MAX_KMEANS_ITERATION_TIMES);

    free(belong);
}

void kmeansII(float *original_data, unsigned int original_size, unsigned int original_dim, float *cluster_set)
{
    init(original_data, original_size, original_dim, cluster_set);
    iteration(original_data, original_size, original_dim, cluster_set);
}