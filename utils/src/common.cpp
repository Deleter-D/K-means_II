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

void belongS2S(size_t *index, float *original_set, float *cluster_set, const int dim, const size_t original_size, const size_t cluster_size)
{
    for (size_t i = 0; i < original_size; i++)
    {
        index[i] = belongV2S(&original_set[i * dim], cluster_set, dim, cluster_size);
    }
}

void kmeanspp(float *cluster_final, float *cluster_set, size_t *omega, size_t k, const int dim, const size_t cluster_size)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, cluster_size - 1);

    size_t index = distrib(gen);

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
        float cost_set2final = costFromS2S(cluster_set, cluster_final, dim, cluster_size, current_k);
        for (size_t i = 0; i < cluster_size; i++)
        {
            // 计算当前向量的概率
            temp_p = omega[i] * costFromV2S(&cluster_set[i * dim], cluster_final, dim, current_k) / cost_set2final;
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
}
void meanVec(float *res, float *original_set, size_t *belong, const int dim, const size_t original_size, const size_t index)
{
    size_t count = 0;
    for (size_t i = 0; i < original_size; i++)
    {
        if (belong[i] == index)
        {
            for (size_t j = 0; j < dim; j++)
            {
                res[j] += original_set[i * dim + j];
            }
            count++;
        }
    }

    for (size_t i = 0; i < dim; i++)
    {
        res[i] /= count;
    }
}

bool isClose(float *cluster_new, float *cluster_old, const int dim, const size_t cluster_size, float epsilon)
{
    for (size_t i = 0; i < cluster_size; i++)
    {
        if (euclideanDistance(&cluster_new[i * dim], &cluster_old[i * dim], dim) > epsilon)
        {
            return false;
        }
    }
    return true;
}

void save(float *data, size_t size, const std::string &filename)
{
    std::ofstream outFile(filename, std::ios::out | std::ios::binary);
    if (!outFile)
    {
        std::cerr << ERROR_HEAD << "Can not open file." << std::endl;
        return;
    }
    outFile.write(reinterpret_cast<const char *>(data), size * sizeof(float));
    outFile.close();
}

void load(float *data, size_t size, const std::string &filename)
{
    std::ifstream inFile(filename, std::ios::in | std::ios::binary);
    if (!inFile)
    {
        std::cerr << ERROR_HEAD << "Can not open file." << std::endl;
        return;
    }
    inFile.read(reinterpret_cast<char *>(data), size * sizeof(float));
    inFile.close();
}

void save(size_t *data, size_t size, const std::string &filename)
{
    std::ofstream outFile(filename, std::ios::out | std::ios::binary);
    if (!outFile)
    {
        std::cerr << ERROR_HEAD << "Can not open file." << std::endl;
        return;
    }
    outFile.write(reinterpret_cast<const char *>(data), size * sizeof(size_t));
    outFile.close();
}

void load(size_t *data, size_t size, const std::string &filename)
{
    std::ifstream inFile(filename, std::ios::in | std::ios::binary);
    if (!inFile)
    {
        std::cerr << ERROR_HEAD << "Can not open file." << std::endl;
        return;
    }
    inFile.read(reinterpret_cast<char *>(data), size * sizeof(size_t));
    inFile.close();
}