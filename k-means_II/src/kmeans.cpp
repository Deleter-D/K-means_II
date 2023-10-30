#include <math.h>
#include <cstring>
#include <random>

#include "../include/kmeans.h"

void kmeans::init()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, original_size - 1);

    size_t index = distrib(gen);

    // 初始状态分配k个聚类中心的内存
    size_t center_bytes = k * original_dim * sizeof(float);
    center = (float *)malloc(center_bytes);
    // 将第一个聚类中心并入集合C
    memcpy(center, &original_data[index], original_dim * sizeof(float));

    float phi = costFromS2S(original_data, center, original_dim, original_size, current_k);
}

void kmeans::iteration() {}

float kmeans::euclideanDistance(float *x, float *y, const int dim)
{
    float sum = 0.0f;
    for (int i = 0; i < dim; i++)
    {
        sum += (x[i] - y[i]) * (x[i] - y[i]);
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