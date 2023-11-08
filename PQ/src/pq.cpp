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

// void build(float *original_data, size_t original_size, int original_dim, unsigned int m)
// {
//     size_t subset_dim = original_dim / m;

//     float **subsets;
//     float **clusters;
//     size_t **indices;
//     subsets = (float **)malloc(m * sizeof(float *));
//     clusters = (float **)malloc(m * sizeof(float *));
//     indices = (size_t **)malloc(m * sizeof(size_t *));
// #pragma omp parallel for
//     for (unsigned int i = 0; i < m; i++)
//     {
//         subsets[i] = (float *)malloc(original_size * subset_dim * sizeof(float));
//         clusters[i] = (float *)malloc(K * subset_dim * sizeof(float));
//         indices[i] = (size_t *)malloc(original_size * sizeof(size_t));
//     }

// // 拆分子集
// #pragma omp parallel for collapse(2)
//     for (size_t i = 0; i < original_size; i++)
//     {
//         for (unsigned int j = 0; j < m; j++)
//         {
//             memcpy(&subsets[j][i * subset_dim], &original_data[i * original_dim + j * subset_dim], subset_dim * sizeof(float));
//         }
//     }

// #pragma omp parallel for
//     for (unsigned int i = 0; i < m; i++)
//     {
//         // 对每个子集进行聚类
//         kmeansII(subsets[i], original_size, subset_dim, clusters[i]);
//         save(clusters[i], original_size * subset_dim, "cluster" + std::to_string(i));
// // 计算每个子集中原始子向量所属的子聚类中心索引
// #ifdef __USE_CUDA__
//         cudaBelongS2S(indices[i], subsets[i], clusters[i], subset_dim, original_size, K);
// #else
//         belongS2S(indices[i], subsets[i], clusters[i], subset_dim, original_size, K);
// #endif
//         save(indices[i], original_size, "index" + std::to_string(i));
//     }
// #pragma omp parallel for
//     for (unsigned int i = 0; i < m; i++)
//     {
//         free(subsets[i]);
//         free(clusters[i]);
//         free(indices[i]);
//     }
//     free(subsets);
//     free(clusters);
//     free(indices);
// }

void query(size_t *result, float *input, size_t original_size, int original_dim, unsigned int m, unsigned int topk)
{
    size_t subvec_dim = original_dim / m;

    float **subvec;
    float **distance_tab;
    float **clusters;
    size_t **indices;
    float **distance;
    subvec = (float **)malloc(m * sizeof(float *));
    distance_tab = (float **)malloc(m * sizeof(float *));
    clusters = (float **)malloc(m * sizeof(float *));
    indices = (size_t **)malloc(m * sizeof(size_t *));
    distance = (float **)malloc(m * sizeof(float *));
#pragma omp parallel for
    for (unsigned int i = 0; i < m; i++)
    {
        subvec[i] = (float *)malloc(subvec_dim * sizeof(float));
        distance_tab[i] = (float *)malloc(K * sizeof(float));
        clusters[i] = (float *)malloc(K * subvec_dim * sizeof(float));
        indices[i] = (size_t *)malloc(original_size * sizeof(size_t));
        distance[i] = (float *)malloc(original_size * sizeof(float));
    }

#pragma omp parallel for
    for (unsigned int i = 0; i < m; i++)
    {
        memcpy(subvec[i], &input[i * subvec_dim], subvec_dim * sizeof(float));
        load(clusters[i], K * subvec_dim, "cluster" + std::to_string(i));
        load(indices[i], original_size, "index" + std::to_string(i));
        memset(distance[i], 0, original_size * sizeof(float));
    }

#ifdef __USE_CUDA__
#pragma omp parallel for
    for (unsigned int i = 0; i < m; i++)
    {
        cudaEuclideanDistance(distance_tab[i], subvec[i], clusters[i], subvec_dim, K);
    }
#else
#pragma omp parallel for collapse(2)
    for (unsigned int i = 0; i < m; i++)
    {

        for (size_t j = 0; j < K; j++)
        {
            distance_tab[i][j] = euclideanDistance(subvec[i], &(clusters[i][j * subvec_dim]), subvec_dim);
        }
    }
#endif

#ifdef __USE_CUDA__
#pragma omp parallel for
    for (unsigned int i = 0; i < m; i++)
    {
        getAsymmetricDistance(distance[i], distance_tab[i], indices[i], original_size);
    }
#else
#pragma omp parallel for collapse(2)
    for (unsigned int i = 0; i < m; i++)
    {
        for (size_t j = 0; j < original_size; j++)
        {
            distance[i][j] = distance_tab[i][indices[i][j]];
        }
    }
#endif

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

#pragma omp parallel for
    for (unsigned int i = 0; i < m; i++)
    {
        free(subvec[i]);
        free(distance_tab[i]);
        free(clusters[i]);
        free(indices[i]);
        free(distance[i]);
    }
    free(subvec);
    free(distance_tab);
    free(clusters);
    free(indices);
    free(distance);
}

void build(float *original_data, size_t original_size, int original_dim, unsigned int m)
{
    float *clusters;
    size_t *indices;

    clusters = (float *)malloc(K * original_dim * sizeof(float));
    indices = (size_t *)malloc(original_size * sizeof(size_t));

    // 对每个子集进行聚类
    kmeansII(original_data, original_size, original_dim, clusters);
    save(clusters, original_size * original_dim, "cluster" + std::to_string(m));
// 计算每个子集中原始子向量所属的子聚类中心索引
#ifdef __USE_CUDA__
    cudaBelongS2S(indices, original_data, clusters, original_dim, original_size, K);
#else
    belongS2S(indices, original_data, clusters, original_dim, original_size, K);
#endif
    save(indices, original_size, "index" + std::to_string(m));

    free(clusters);
    free(indices);
}

void productQuantization(std::string &filename, size_t original_size, int original_dim, unsigned int m)
{
    split_file(filename, original_size, original_dim, m);

    size_t subset_dim = original_dim / m;

#pragma omp parallel for
    for (unsigned int i = 0; i < m; i++)
    {
        int fd = open(("subset" + std::to_string(m)).c_str(), O_RDONLY);
        struct stat sb;
        bool error_flag = false;

        if (fd == -1)
        {
            std::cerr << ERROR_HEAD << "Can not open file." << std::endl;
            error_flag = true;
        }

        if (!error_flag || fstat(fd, &sb) == -1)
        {
            std::cerr << ERROR_HEAD << "Can not get file size." << std::endl;
            error_flag = true;
        }

        float *mapped_data;
        if (!error_flag)
        {
            mapped_data = (float *)mmap(NULL, sb.st_size, PROT_READ, MAP_SHARED, fd, 0);
        }

        if (!error_flag || mapped_data == MAP_FAILED)
        {
            close(fd);
            std::cerr << ERROR_HEAD << "Can not map file to memory." << std::endl;
            error_flag = true;
        }

        if (!error_flag)
        {
            build(mapped_data, original_size, subset_dim, i);
        }

        if (!error_flag || munmap(mapped_data, sb.st_size) == -1)
        {
            std::cerr << ERROR_HEAD << "Can not unmap file from memory." << std::endl;
        }

        close(fd);
    }
}