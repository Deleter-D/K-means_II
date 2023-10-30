#include <math.h>
#include <cstring>
#include <random>
#include <functional>
#include <map>
#include <set>
#include <iostream>
#include <cmath>

#include "../include/kmeans.h"

void kmeans::init()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, original_size - 1);

    size_t index = distrib(gen);

    // 分配2 * k * ITERATION_TIMES个聚类中心的内存
    size_t center_bytes = (2 * k * ITERATION_TIMES + 1) * original_dim * sizeof(float);
    center = (float *)malloc(center_bytes);
    memset(center, 0, center_bytes);

    // 记录聚类中心向量的全局索引
    std::set<size_t> center_index;

    // 将第一个聚类中心并入集合C
    memcpy(center, &original_data[index], original_dim * sizeof(float));
    current_k = 1;
    center_index.insert(index); // 记录第一个聚类中心全局索引

    // 计算此时聚类中心集与全集的代价
    float phi = costFromS2S(original_data, center, original_dim, original_size, current_k);

    // 迭代
    for (int i = 0; i < ITERATION_TIMES; i++)
    {
        // 存放概率值和索引的key-value对
        std::multimap<float, size_t, std::greater<float>> probability;
        float current_p;
        for (int j = 0; j < original_size; j++)
        {
            float *temp = &original_data[j * original_dim];
            // 计算当前向量的概率
            current_p = 2 * k * costFromV2S(temp, center, original_dim, current_k) /
                        costFromS2S(original_data, center, original_dim, original_size, current_k);
            probability.insert({current_p, j});
        }
        auto beg = probability.cbegin();
        auto end = probability.cend();
        // 将map中的前l个向量并入聚类中心集
        for (int i = 0; i < 2 * k; i++)
        {
            auto temp_index = beg->second;
            if (beg != end && !center_index.count(temp_index))
            {
                memcpy(&center[original_dim * current_k], &original_data[temp_index * original_dim], original_dim * sizeof(float));
                current_k++;                     // 聚类中心当前数量自增
                center_index.insert(temp_index); // 记录并入聚类中心集的全局索引
            }
            beg++;
        }
        probability.clear();
    }
    std::cout << center_index.size() << std::endl;
}

void kmeans::iteration() {}

float kmeans::euclideanDistance(float *x, float *y, const int dim)
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

float kmeans::costFromV2S(float *x, float *set, const int dim, const int size)
{
    float min = MAXFLOAT;
    float temp = MAXFLOAT;
    for (int i = 0; i < size; i++)
    {
        temp = euclideanDistance(x, &set[i * dim], dim);
        if (temp < min)
            min = temp;
    }
    return min;
}

float kmeans::costFromS2S(float *set1, float *set2, const int dim, const int size1, const int size2)
{
    float sum = 0.0f;
    for (int i = 0; i < size1; i++)
    {
        sum += costFromV2S(&set1[i * dim], set2, dim, size2);
    }
    return sum;
}