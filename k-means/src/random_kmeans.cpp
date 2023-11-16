#include <random>
#include <omp.h>
#include <cstring>
#include "../include/random_kmeans.h"

#define __USE_CUDA__

void randomInit(float *original_data, size_t original_size, size_t original_dim, float *cluster_set, int cluster_size)
{
#ifdef DEBUG
    time_t start_time = 0;
    time_t end_time = 0;
    start_time = time(NULL);
#endif

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, original_size - 1);

    size_t vec_bytes = original_dim * sizeof(float);
    size_t cluster_bytes = cluster_size * vec_bytes;
    size_t current_k = 6 * cluster_size;

    // 随即选取6k个向量
    size_t *indices = (size_t *)malloc(current_k * sizeof(size_t));
#pragma omp parallel for
    for (int i = 0; i < current_k; i++)
    {
        indices[i] = distrib(gen);
    }

    float *cluster_set_temp = (float *)malloc(current_k * vec_bytes);
    memset(cluster_set_temp, 0, current_k * vec_bytes);

#pragma omp parallel for
    for (int i = 0; i < current_k; i++)
    {
        memcpy(&cluster_set_temp[i * original_dim], &original_data[indices[i] * original_dim], vec_bytes);
    }
    free(indices);

#ifdef DEBUG
    std::cout << DEBUG_HEAD << "initial clusters generated.\n";
    std::cout << DEBUG_HEAD << "begin to calculate omega.\n";
#endif

    // 记录每个聚类中心的权重
    size_t *omega = (size_t *)malloc(current_k * sizeof(size_t));
    memset(omega, 0, current_k * sizeof(size_t));

    // 记录每个向量归属的聚类中心索引
    size_t *index_X2C = (size_t *)malloc(original_size * sizeof(size_t));

#ifdef __USE_CUDA__
    cudaBelongS2S(index_X2C, original_data, cluster_set_temp, original_dim, original_size, current_k);
#else
    belongS2S(index_X2C, original_data, cluster_set_temp, original_dim, original_size, current_k);
#endif

#pragma omp parallel for
    for (int i = 0; i < original_size; i++)
    {
        size_t index = index_X2C[i];
#pragma omp atomic
        omega[index]++;
    }

    free(index_X2C);

#ifdef DEBUG
    std::cout << DEBUG_HEAD << "omega calculation finished.\n";
    std::cout << DEBUG_HEAD << "begin to K-means++\n";
#endif

    float *cluster_final = (float *)malloc(cluster_bytes);
#ifdef __USE_CUDA__
    cudaKmeanspp(cluster_final, cluster_set_temp, omega, cluster_size, original_dim, current_k);
#else
    kmeanspp(cluster_final, cluster_set_temp, omega, cluster_size, original_dim, current_k);
#endif
    free(omega);
    free(cluster_set_temp);

#ifdef DEBUG
    std::cout << DEBUG_HEAD << "K-means++ finished.\n";
#endif

    memcpy(cluster_set, cluster_final, cluster_bytes);
    free(cluster_final);

#ifdef DEBUG
    std::cout << DEBUG_HEAD << "intilization finished.\n";
#endif

#ifdef DEBUG
    end_time = time(NULL);
    std::cout << DEBUG_HEAD << "elapsed time: " << (end_time - start_time) << "\n";
#endif
}

void randomIteration(float *original_data, size_t original_size, size_t original_dim, float *cluster_set, int cluster_size)
{
    size_t vec_bytes = original_dim * sizeof(float);
    size_t cluster_bytes = cluster_size * vec_bytes;

    // 计算全集中的向量所属的聚类中心的索引
    size_t *belong = (size_t *)malloc(original_size * sizeof(size_t));
#ifdef __USE_CUDA__
    cudaBelongS2S(belong, original_data, cluster_set, original_dim, original_size, cluster_size);
#else
    belongS2S(belong, original_data, cluster_set, original_dim, original_size, cluster_size);
#endif

    float *cluster_new = (float *)malloc(cluster_bytes);
    memcpy(cluster_new, cluster_set, cluster_bytes);

    int iteration_times = 0;
    bool isclose;

    size_t *temp_count = (size_t *)malloc(cluster_size * sizeof(size_t)); // 删除
    do
    {
        memcpy(cluster_set, cluster_new, cluster_bytes);

#ifdef __USE_CUDA__
        cudaGetNewCluster(cluster_new, original_data, belong, original_dim, original_size);
#else
        // 产生新的聚类中心集
        float *mean_vec = (float *)malloc(vec_bytes);
        memset(mean_vec, 0, vec_bytes);
        for (int i = 0; i < cluster_size; i++)
        {
            memset(mean_vec, 0, vec_bytes);
            meanVec(mean_vec, original_data, belong, original_dim, original_size, i);
            memcpy(&cluster_new[i * original_dim], mean_vec, vec_bytes);
        }
        free(mean_vec);
#endif

#ifdef __USE_CUDA__
        cudaBelongS2S(belong, original_data, cluster_new, original_dim, original_size, cluster_size);
#else
        // 更新归属关系索引
        belongS2S(belong, original_data, cluster_new, original_dim, original_size, cluster_size);
#endif

#ifdef DEBUG
        printf("迭代%d\n", iteration_times);
        memset(temp_count, 0, cluster_size * sizeof(size_t));

        for (int i = 0; i < original_size; i++)
        {
            size_t index = belong[i];

            temp_count[index]++;
        }

        for (int i = 0; i < cluster_size; i++)
        {
            printf("%d: %ld个\n", i, temp_count[i]);
        }
#endif

        iteration_times++;

#ifdef __USE_CUDA__
        isclose = cudaIsClose(cluster_new, cluster_set, original_dim, cluster_size, THRESHOLD);
#else
        isclose = isClose(cluster_new, cluster_set, original_dim, cluster_size, THRESHOLD);
#endif

#ifdef DEBUG
        printf("迭代结果%d: %s\n", iteration_times, isclose ? "close" : "not close");
#endif

    } while (!isclose && iteration_times < MAX_KMEANS_ITERATION_TIMES);

    free(belong);
}

void randomKmeans(float *original_data, size_t original_size, size_t original_dim, float *cluster_set, int cluster_size)
{
    randomInit(original_data, original_size, original_dim, cluster_set, cluster_size);
    randomIteration(original_data, original_size, original_dim, cluster_set, cluster_size);
}