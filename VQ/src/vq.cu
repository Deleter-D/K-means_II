#include <algorithm>
#include <omp.h>
#include <vector>
#include <utility>
#include <iostream>
#include "../include/vq.cuh"

void VQBuild(float *original_data, unsigned int original_size, int original_dim, char *prefix)
{
    std::string prefix_str = prefix;
    size_t cluster_0_bytes = K0 * original_dim * sizeof(float);
    size_t indices_0_bytes = original_size * sizeof(unsigned int);

    float *cluster_0 = (float *)malloc(cluster_0_bytes);
#ifdef DEBUG
    std::cout << DEBUG_HEAD << "begining to first level K-means.\n";
#endif
    randomKmeans(original_data, original_size, original_dim, cluster_0, K0);
#ifdef DEBUG
    std::cout << DEBUG_HEAD << "first level K-means finished.\n";
#endif

    unsigned int *indices_0 = (unsigned int *)malloc(indices_0_bytes);
#ifdef DEBUG
    std::cout << DEBUG_HEAD << "begining to calculate first level belong.\n";
#endif
    cudaBelongS2S(indices_0, original_data, cluster_0, original_dim, original_size, K0);
#ifdef DEBUG
    std::cout << DEBUG_HEAD << "first level belong finished.\n";
#endif
    unsigned int *catagories_count = (unsigned int *)malloc(K0 * sizeof(unsigned int));

#pragma omp parallel for
    for (int i = 0; i < original_size; i++)
    {
        unsigned int temp = indices_0[i];
#pragma omp atomic
        catagories_count[temp]++;
    }

    save(cluster_0, cluster_0_bytes, prefix_str + "cluster_0");
    save(indices_0, indices_0_bytes, prefix_str + "indices_0");
    save(catagories_count, K0 * sizeof(unsigned int), prefix_str + "catagories");
#ifdef DEBUG
    std::cout << DEBUG_HEAD << "file of first level saved.\n";
#endif

    free(catagories_count);

    size_t cluster_1_bytes = K1 * original_dim * sizeof(float);
    size_t indices_1_bytes = K0 * sizeof(unsigned int);

    float *cluster_1 = (float *)malloc(cluster_1_bytes);
#ifdef DEBUG
    std::cout << DEBUG_HEAD << "begining to second level K-means.\n";
#endif
    randomKmeans(cluster_0, K0, original_dim, cluster_1, K1);
#ifdef DEBUG
    std::cout << DEBUG_HEAD << "second level K-means finished.\n";
#endif

    unsigned int *indices_1 = (unsigned int *)malloc(indices_1_bytes);
#ifdef DEBUG
    std::cout << DEBUG_HEAD << "begining to calculate second level belong.\n";
#endif
    cudaBelongS2S(indices_1, cluster_0, cluster_1, original_dim, K0, K1);
#ifdef DEBUG
    std::cout << DEBUG_HEAD << "second level belong finished.\n";
#endif

    save(cluster_1, cluster_1_bytes, prefix_str + "cluster_1");
    save(indices_1, indices_1_bytes, prefix_str + "indices_1");
#ifdef DEBUG
    std::cout << DEBUG_HEAD << "file of second level saved.\n";
#endif

    free(cluster_0);
    free(indices_0);
    free(cluster_1);
    free(indices_1);
}

__global__ void levelQueryKernel(unsigned int *result, float *query_set, float *cluster, unsigned int *indices, unsigned int *belong,
                                 unsigned int dim, unsigned int query_size, unsigned int cluster_size, unsigned int indices_size)
{
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    if (idx < query_size)
    {
        float min_distance = INFINITY;
        float temp;

        for (int i = 0; i < cluster_size; i++)
        {
            if (indices[i] == belong[idx])
            {
                float dist = 0.0f;
                for (int j = 0; j < dim; j++)
                {
                    float diff = query_set[idx * dim + j] - cluster[i * dim + j];
                    dist += diff * diff;
                }
                temp = dist;

                if (temp < min_distance)
                {
                    min_distance = temp;

                    result[idx] = i;
                }
            }
        }
    }
}

void VQQuery(unsigned int *result, float *query_set, unsigned int query_size, float *original_data, unsigned int original_dim, unsigned int original_size, unsigned int topk,
           float *cluster_0, float *cluster_1, unsigned int *indices_0, unsigned int *indices_1, unsigned int *catagories_count)
{
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 常驻设备
    size_t cluster_0_bytes = K0 * original_dim * sizeof(float);
    size_t indices_0_bytes = original_size * sizeof(unsigned int);
    size_t cluster_1_bytes = K1 * original_dim * sizeof(float);
    size_t indices_1_bytes = K0 * sizeof(unsigned int);
    size_t query_set_bytes = query_size * original_dim * sizeof(float);
    float *d_cluster_0, *d_cluster_1, *d_query_set;
    unsigned int *d_indices_0, *d_indices_1;
    cudaMalloc((void **)&d_cluster_0, cluster_0_bytes);
    cudaMalloc((void **)&d_cluster_1, cluster_1_bytes);
    cudaMalloc((void **)&d_indices_0, indices_0_bytes);
    cudaMalloc((void **)&d_indices_1, indices_1_bytes);
    cudaMalloc((void **)&d_query_set, query_set_bytes);
    cudaMemcpy(d_cluster_0, cluster_0, cluster_0_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cluster_1, cluster_1, cluster_1_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices_0, indices_0, indices_0_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices_1, indices_1, indices_1_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_query_set, query_set, query_set_bytes, cudaMemcpyHostToDevice);

    // 第一层查询    // float **candidate = (float **)malloc(query_size * sizeof(float *));

    size_t belong_bytes = query_size * sizeof(unsigned int);
    unsigned int *d_belong_1;
    cudaMalloc((void **)&d_belong_1, belong_bytes);

    dim3 block(1024);
    dim3 grid((query_size + block.x - 1) / block.x);

    belongS2SKernel<<<grid, block>>>(d_belong_1, d_query_set, d_cluster_1, original_dim, query_size, K1);

    cudaStreamSynchronize(stream);

    // 第二层查询
    unsigned int *belong_0;
    belong_0 = (unsigned int *)malloc(belong_bytes * sizeof(unsigned int));
    unsigned int *d_belong_0;
    cudaMalloc((void **)&d_belong_0, belong_bytes);

    levelQueryKernel<<<grid, block>>>(d_belong_0, d_query_set, d_cluster_0, d_indices_1, d_belong_1, original_dim, query_size, K0, K0);

    cudaStreamSynchronize(stream);
    cudaMemcpy(belong_0, d_belong_0, belong_bytes, cudaMemcpyDeviceToHost);

    // 第三层查询
    cudaStream_t *streams = (cudaStream_t *)malloc(query_size * sizeof(cudaStream_t));

#pragma omp parallel for
    for (int i = 0; i < query_size; i++)
    {
        cudaStreamCreate(&streams[i]);
        int count = 0;
        size_t candidate_bytes = catagories_count[i] * original_dim * sizeof(float);
        size_t distance_bytes = catagories_count[i] * sizeof(float);

        float *distance = (float *)malloc(distance_bytes);

        std::vector<std::pair<float, unsigned int>> dist_idx(catagories_count[i]);

        float *d_candidate, *d_distance;
        cudaMalloc((void **)&d_candidate, candidate_bytes);
        cudaMalloc((void **)&d_distance, distance_bytes);

        cudaMemset(d_distance, 0, distance_bytes);
        for (int j = 0; j < original_size; j++)
        {
            if (indices_0[j] == belong_0[i])
            {
                cudaMemcpy(&d_candidate[count * original_dim], &original_data[j * original_dim], original_dim * sizeof(float), cudaMemcpyHostToDevice);
                dist_idx[count].second = j;
                count++;
            }
        }

        dim3 block1(original_dim);
        dim3 grid1(catagories_count[i]);
        euclideanDistanceKernel<<<grid1, block1>>>(d_distance, &d_query_set[i * original_dim], d_candidate, original_dim, catagories_count[i]);

        cudaMemcpy(distance, d_distance, distance_bytes, cudaMemcpyDeviceToHost);
        cudaStreamSynchronize(streams[i]);

#pragma omp parallel for
        for (int j = 0; j < catagories_count[i]; j++)
        {
            dist_idx[j].first = distance[j];
        }

        std::partial_sort(dist_idx.begin(), dist_idx.begin() + topk, dist_idx.end(), std::greater<std::pair<float, unsigned int>>());

#pragma omp parallel for
        for (int j = 0; j < topk; j++)
        {
            result[i * topk + j] = dist_idx[j].second;
        }
    }
}