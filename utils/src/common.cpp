#include <math.h>
#include <random>
#include <cstring>
#include "../include/common.h"

float euclideanDistance(float *x, float *y, const int dim)
{
    float sum = 0.0f;
    float difference = 0.0f;
    for (int i = 0; i < dim; i++)
    {
        difference = x[i] - y[i];
        sum += difference * difference;
    }
    return sum;
}

float costFromV2S(float *x, float *cluster_set, const int dim, const size_t size)
{
    float min = MAXFLOAT;
    float temp = MAXFLOAT;
    for (size_t i = 0; i < size; i++)
    {
        temp = euclideanDistance(x, &cluster_set[i * dim], dim);
        if (temp < min)
            min = temp;
    }
    return min;
}

float costFromS2S(float *original_set, float *cluster_set, const int dim, const size_t original_size, const size_t cluster_size)
{
    float sum = 0.0f;
    for (size_t i = 0; i < original_size; i++)
    {
        sum += costFromV2S(&original_set[i * dim], cluster_set, dim, cluster_size);
    }
    return sum;
}

size_t belongV2S(float *x, float *cluster_set, const int dim, const size_t size)
{
    float min = MAXFLOAT;
    float temp;
    size_t index;
    for (size_t i = 0; i < size; i++)
    {
        temp = euclideanDistance(x, &cluster_set[i * dim], dim);
        if (temp < min)
        {
            min = temp;
            index = i;
        }
    }
    return index;
}

size_t *belongS2S(float *original_set, float *cluster_set, const int dim, const size_t original_size, const size_t cluster_size)
{
    size_t *index = (size_t *)malloc(original_size * sizeof(size_t));
    for (size_t i = 0; i < original_size; i++)
    {
        index[i] = belongV2S(&original_set[i * dim], cluster_set, dim, cluster_size);
    }
    return index;
}

float *kmeanspp(float *cluster_set, size_t *omega, size_t k, const int dim, const size_t cluster_size)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, cluster_size - 1);

    size_t index = distrib(gen);

    // 申请最终聚类中心集的内存
    float *cluster_final = (float *)malloc(k * dim * sizeof(float));
    // 均匀分布中随机采样一个原聚类中心集的向量放入最终聚类中心集中
    memcpy(&cluster_final[0], &cluster_set[index * dim], dim * sizeof(float));
    size_t current_k = 1;

    float max_p;
    float temp_p;
    size_t max_p_index;

    // 迭代k-1次，每次取一个聚类中心进入c_final
    while (current_k < k)
    {
        max_p = -1.0f;
        for (size_t i = 0; i < cluster_size; i++)
        {
            // 计算当前向量的概率
            temp_p = omega[i] * costFromV2S(&cluster_set[i * dim], cluster_final, dim, current_k) /
                     costFromS2S(cluster_set, cluster_final, dim, cluster_size, current_k);
            // 记录概率最大的向量信息
            if (temp_p > max_p)
            {
                max_p = temp_p;
                max_p_index = i;
            }
        }
        // 将概率最大的向量并入最终聚类中心集
        memcpy(&cluster_final[current_k * dim], &cluster_set[max_p_index * dim], dim * sizeof(float));
        current_k++;
    }

    return cluster_final;
}