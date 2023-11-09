#include <algorithm>
#include <iostream>
#include <stdlib.h>
#include <cstring>
#include <string>
#include <omp.h>
#include <cmath>
#include <vector>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include "../include/pq.h"
#include "../../k-means_II/include/kmeans_II.h"

#define __USE_CUDA__

void build(float *original_data, size_t original_size, int dim, unsigned int m)
{
    float *clusters;
    size_t *indices;

    clusters = (float *)malloc(K * dim * sizeof(float));
    indices = (size_t *)malloc(original_size * sizeof(size_t));

    // 对每个子集进行聚类
    kmeansII(original_data, original_size, dim, clusters);
    save(clusters, K * dim, "cluster" + std::to_string(m));
// 计算每个子集中原始子向量所属的子聚类中心索引
#ifdef __USE_CUDA__
    cudaBelongS2S(indices, original_data, clusters, dim, original_size, K);
#else
    belongS2S(indices, original_data, clusters, dim, original_size, K);
#endif
    save(indices, original_size, "index" + std::to_string(m));

    free(clusters);
    free(indices);
}

void subQuery(float *distance, float *input, float *cluster, size_t *index, size_t original_size, int sub_dim)
{
    float *distance_tab;
    distance_tab = (float *)malloc(K * sizeof(float));

#ifdef __USE_CUDA__
    cudaEuclideanDistance(distance_tab, input, cluster, sub_dim, K);
#else
#pragma omp parallel for
    for (unsigned int i = 0; i < K; i++)
    {
        distance_tab[i] = euclideanDistance(input, &cluster[i], sub_dim);
    }
#endif

#ifdef __USE_CUDA__
    cudaGetAsymmetricDistance(distance, distance_tab, index, original_size);
#else
#pragma omp parallel for
    for (size_t i = 0; i < original_size; i++)
    {
        distance[i] = distance_tab[index[i]];
    }
#endif
    free(distance_tab);
}

void query(size_t *result, float *input, float **clusters, size_t **indices, size_t original_size, int original_dim, unsigned int m, unsigned int topk)
{
    float **input_sub, **distance;
    input_sub = (float **)malloc(m * sizeof(float *));
    distance = (float **)malloc(m * sizeof(float *));

    size_t sub_dim = original_dim / m;

#pragma omp parallel for
    for (unsigned int i = 0; i < m; i++)
    {
        input_sub[i] = &input[i * sub_dim];
        distance[i] = (float *)malloc(original_size * sizeof(float));
    }

#pragma omp parallel for
    for (unsigned int i = 0; i < m; i++)
    {
        subQuery(distance[i], input_sub[i], clusters[i], indices[i], original_size, sub_dim);
    }

    std::vector<std::pair<float, size_t>> distance_final(original_size, {0, 0});

#pragma omp parallel for
    for (size_t i = 0; i < original_size; i++)
    {
        for (unsigned int j = 0; j < m; j++)
        {
            distance_final[i].first += distance[j][i];
        }
        distance_final[i].second = i;
    }

    std::partial_sort(distance_final.begin(), distance_final.begin() + topk, distance_final.end(), std::greater<std::pair<float, size_t>>());

#pragma omp parallel for
    for (unsigned int i = 0; i < topk; i++)
    {
        result[i] = distance_final[i].second;
    }
}

void productQuantizationBuild(std::string &filename, size_t original_size, int original_dim, unsigned int m)
{
    split_file(filename, original_size, original_dim, m);

    size_t subset_dim = original_dim / m;

#pragma omp parallel for
    for (unsigned int i = 0; i < m; i++)
    {
        int fd = open(("subset" + std::to_string(i)).c_str(), O_RDONLY);
        struct stat sb;
        bool error_flag = false;

        if (fd == -1)
        {
            std::cerr << ERROR_HEAD << "Can not open file." << std::endl;
            error_flag = true;
        }

        if (error_flag || fstat(fd, &sb) == -1)
        {
            std::cerr << ERROR_HEAD << "Can not get file size." << std::endl;
            error_flag = true;
        }

        float *mapped_data;
        if (!error_flag)
        {
            mapped_data = (float *)mmap(NULL, sb.st_size, PROT_READ, MAP_SHARED, fd, 0);
            if (mapped_data == MAP_FAILED)
            {
                close(fd);
                std::cerr << ERROR_HEAD << "Can not map file to memory." << std::endl;
                error_flag = true;
            }
        }

        if (!error_flag)
        {
            build(mapped_data, original_size, subset_dim, i);
            if (munmap(mapped_data, sb.st_size) == -1)
            {
                std::cerr << ERROR_HEAD << "Can not unmap file from memory." << std::endl;
            }
        }

        close(fd);
    }
}

void productQuantizationQuery(size_t *result, float *input, size_t input_size, size_t original_size, int orininal_dim, unsigned int m, unsigned int topk)
{
    float **clusters = (float **)malloc(m * sizeof(float *));
    size_t **indices = (size_t **)malloc(m * sizeof(size_t *));
    size_t sub_dim = orininal_dim / m;
#pragma omp parallel for
    for (int i = 0; i < m; i++)
    {
        clusters[i] = (float *)malloc(K * sub_dim * sizeof(float));
        indices[i] = (size_t *)malloc(original_size * sizeof(size_t));
        load(clusters[i], K * sub_dim, "cluster" + std::to_string(i));
        load(indices[i], original_size, "index" + std::to_string(i));
    }

#pragma omp parallel for
    for (size_t i = 0; i < input_size; i++)
    {
        query(&result[i * topk], &input[i * orininal_dim], clusters, indices, original_size, orininal_dim, m, topk);
    }

#pragma omp parallel for
    for (int i = 0; i < m; i++)
    {
        free(clusters[i]);
        free(indices[i]);
    }
    free(clusters);
    free(indices);
}