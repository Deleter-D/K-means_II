#include "../include/common.cuh"
#include "../include/config.h"
#include <stdio.h>
#include <random>
#include <omp.h>
#include <math.h>
#include <iostream>

__global__ void euclideanDistanceKernel(float *distance, float *vec, float *set, const int dim, const int size)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + tid;

    extern __shared__ float temp[];

    // 计算每对元素差的平方，存入temp
    if (idx < dim * size)
    {
        temp[tid] = (vec[tid] - set[idx]) * (vec[tid] - set[idx]);
    }
    __syncthreads();

    if (tid < 32)
        temp[tid] = temp[tid] + temp[tid + 32] + temp[tid + 64];
    __syncthreads();

    // 交错匹配的归约求和
    for (int stride = 16; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            temp[tid] += temp[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
        distance[blockIdx.x] = temp[0];
}

void cudaEuclideanDistance(float *distance, float *vec, float *set, const int dim, const int size)
{
    size_t MAX_SIZE = 8e9 / (dim * sizeof(float));
    int iter_times = (size / MAX_SIZE) + 1;
    int size_per_iter;
    int size_last_iter;
    if (iter_times == 1)
    {
        size_per_iter = size;
        size_last_iter = size;
    }
    else
    {
        size_per_iter = MAX_SIZE;
        size_last_iter = size - MAX_SIZE * (iter_times - 1);
    }

    cudaStream_t *stream = (cudaStream_t *)malloc(iter_times * sizeof(cudaStream_t));
    size_t vec_bytes = dim * sizeof(float);
    float **d_distances, **d_vecs, **d_sets;
    d_distances = (float **)malloc(iter_times * sizeof(float *));
    d_vecs = (float **)malloc(iter_times * sizeof(float *));
    d_sets = (float **)malloc(iter_times * sizeof(float *));

    for (int i = 0; i < iter_times; i++)
    {
        cudaStreamCreate(&stream[i]);

        size_t distance_bytes;
        size_t set_bytes;
        size_t current_size;

        if (i == iter_times - 1)
        {
            distance_bytes = size_last_iter * sizeof(float);
            set_bytes = dim * static_cast<size_t>(size_last_iter) * sizeof(float);
            current_size = size_last_iter;
        }
        else
        {
            distance_bytes = size_per_iter * sizeof(float);
            set_bytes = dim * static_cast<size_t>(size_per_iter) * sizeof(float);
            current_size = size_per_iter;
        }

        cudaMalloc((void **)&d_distances[i], distance_bytes);
        cudaMalloc((void **)&d_vecs[i], vec_bytes);
        cudaMalloc((void **)&d_sets[i], set_bytes);

        cudaMemcpy(d_vecs[i], vec, vec_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_sets[i], &set[i * dim * size_per_iter], set_bytes, cudaMemcpyHostToDevice);
        cudaMemset(d_distances[i], 0, distance_bytes);

        dim3 block(dim);
        dim3 grid((current_size * dim + block.x - 1) / block.x);
        euclideanDistanceKernel<<<grid, block, dim * sizeof(float), stream[i]>>>(d_distances[i], d_vecs[i], d_sets[i], dim, current_size);

        cudaMemcpy(&distance[i * size_per_iter], d_distances[i], distance_bytes, cudaMemcpyDeviceToHost);
    }

    for (int i = 0; i < iter_times; i++)
    {
        cudaStreamSynchronize(stream[i]);

        cudaFree(d_distances[i]);
        cudaFree(d_vecs[i]);
        cudaFree(d_sets[i]);

        cudaStreamDestroy(stream[i]);
    }
    free(stream);
    free(d_distances);
    free(d_vecs);
    free(d_sets);
}

float cudaCostFromV2S(float *vec, float *cluster_set, const int dim, const unsigned int size)
{
    float min = 3.40282347e+38F;

    float *distance = (float *)malloc(size * sizeof(float));
    cudaEuclideanDistance(distance, vec, cluster_set, dim, size);

#pragma omp parallel for private(i) reduction(min : min)
    for (unsigned int i = 0; i < size; i++)
    {
        if (distance[i] < min)
            min = distance[i];
    }

    free(distance);

    return min;
}

__global__ void costFromS2SKernel(float *distances, float *original_set, float *cluster_set, int dim, unsigned int original_size, unsigned int cluster_size)
{
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.y * gridDim.x + blockIdx.x;
    unsigned int idx = bid * blockDim.x + tid;

    extern __shared__ float distance_temp[];

    if (idx < original_size)
    {
        distances[idx] = INFINITY;

        for (int j = 0; j < cluster_size; j++)
        {
            float dist = 0.0f;
            for (int k = 0; k < dim; k++)
            {
                float diff = original_set[idx * dim + k] - cluster_set[j * dim + k];
                dist += diff * diff;
            }
            distance_temp[tid] = dist;

            if (distance_temp[tid] < distances[idx])
            {
                distances[idx] = distance_temp[tid];
            }
        }
    }
}

float cudaCostFromS2S(float *original_set, float *cluster_set, const int dim, const unsigned int original_size, const unsigned int cluster_size)
{
    float *distances = (float *)malloc(original_size * sizeof(float));

    size_t MAX_SIZE = 2e10 / (dim * sizeof(float));
    int iter_times = (original_size / MAX_SIZE) + 1;
    int size_per_iter;
    int size_last_iter;
    if (iter_times == 1)
    {
        size_per_iter = original_size;
        size_last_iter = original_size;
    }
    else
    {
        size_per_iter = MAX_SIZE;
        size_last_iter = original_size - MAX_SIZE * (iter_times - 1);
    }

    cudaStream_t *stream = (cudaStream_t *)malloc(iter_times * sizeof(cudaStream_t));
    float **d_original_sets, **d_distances;
    d_original_sets = (float **)malloc(iter_times * sizeof(float *));
    d_distances = (float **)malloc(iter_times * sizeof(float *));

    float *d_cluster_set;
    cudaMalloc((void **)&d_cluster_set, cluster_size * dim * sizeof(float));
    cudaMemcpy(d_cluster_set, cluster_set, cluster_size * dim * sizeof(float), cudaMemcpyHostToDevice);

    for (int i = 0; i < iter_times; i++)
    {
        cudaStreamCreate(&stream[i]);

        size_t distance_bytes;
        size_t set_bytes;
        unsigned int current_size;

        if (i == iter_times - 1)
        {
            distance_bytes = size_last_iter * sizeof(float);
            set_bytes = dim * static_cast<size_t>(size_last_iter) * sizeof(float);
            current_size = size_last_iter;
        }
        else
        {
            distance_bytes = size_per_iter * sizeof(float);
            set_bytes = dim * static_cast<size_t>(size_per_iter) * sizeof(float);
            current_size = size_per_iter;
        }

        cudaMalloc((void **)&d_original_sets[i], set_bytes);
        cudaMalloc((void **)&d_distances[i], distance_bytes);

        cudaMemcpy(d_original_sets[i], &original_set[i * dim * size_per_iter], set_bytes, cudaMemcpyHostToDevice);
        cudaMemset(d_distances[i], 0, distance_bytes);

        dim3 block(1024);
        unsigned int grid_dim = ceil(sqrt((current_size + block.x - 1) / block.x));
        dim3 grid(grid_dim, grid_dim);

        costFromS2SKernel<<<grid, block, block.x * sizeof(float), stream[i]>>>(d_distances[i], d_original_sets[i], d_cluster_set, dim, current_size, cluster_size);

        cudaMemcpy(&distances[i * size_per_iter], d_distances[i], distance_bytes, cudaMemcpyDeviceToHost);

        auto err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            std::cout << cudaGetErrorString(err) << "\n";
        }
    }

    for (int i = 0; i < iter_times; i++)
    {
        cudaStreamSynchronize(stream[i]);

        cudaFree(d_original_sets[i]);
        cudaFree(d_distances[i]);

        cudaStreamDestroy(stream[i]);
    }
    cudaFree(d_cluster_set);
    free(stream);
    free(d_original_sets);
    free(d_distances);

    float sum = 0.0f;
#pragma omp parallel for reduction(+ : sum)
    for (unsigned int i = 0; i < original_size; i++)
    {
        sum += distances[i];
    }

    free(distances);

    return sum;
}

unsigned int cudaBelongV2S(float *x, float *cluster_set, const int dim, const unsigned int size)
{
    float min = 3.40282347e+38F;
    unsigned int index;

    float *distance = (float *)malloc(size * sizeof(float));
    cudaEuclideanDistance(distance, x, cluster_set, dim, size);

#pragma omp parallel for private(i) reduction(min : min)
    for (unsigned int i = 0; i < size; i++)
    {
        if (distance[i] < min)
        {
            min = distance[i];
            index = i;
        }
    }

    free(distance);

    return index;
}

__global__ void belongS2SKernel(unsigned int *indices, float *original_set, float *cluster_set, int dim, unsigned int original_size, unsigned int cluster_size)
{
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.y * gridDim.x + blockIdx.x;
    unsigned int idx = bid * blockDim.x + tid;

    if (idx < original_size)
    {
        float min_distance = INFINITY;
        float temp;

        for (int i = 0; i < cluster_size; i++)
        {
            float dist = 0.0f;
            for (int j = 0; j < dim; j++)
            {
                float diff = original_set[idx * dim + j] - cluster_set[i * dim + j];
                dist += diff * diff;
            }
            temp = dist;

            if (temp < min_distance)
            {
                min_distance = temp;

                indices[idx] = i;
            }
        }
    }
}

void cudaBelongS2S(unsigned int *index, float *original_set, float *cluster_set, const int dim, const unsigned int original_size, const unsigned int cluster_size)
{
    size_t MAX_SIZE = 2e10 / (dim * sizeof(float));
    int iter_times = (original_size / MAX_SIZE) + 1;
    int size_per_iter;
    int size_last_iter;
    if (iter_times == 1)
    {
        size_per_iter = original_size;
        size_last_iter = original_size;
    }
    else
    {
        size_per_iter = MAX_SIZE;
        size_last_iter = original_size - MAX_SIZE * (iter_times - 1);
    }

    cudaStream_t *stream = (cudaStream_t *)malloc(iter_times * sizeof(cudaStream_t));
    float **d_original_sets;
    unsigned int **d_indices;
    d_original_sets = (float **)malloc(iter_times * sizeof(float *));
    d_indices = (unsigned int **)malloc(iter_times * sizeof(unsigned int *));

    float *d_cluster_set;
    cudaMalloc((void **)&d_cluster_set, cluster_size * dim * sizeof(float));
    cudaMemcpy(d_cluster_set, cluster_set, cluster_size * dim * sizeof(float), cudaMemcpyHostToDevice);

    for (int i = 0; i < iter_times; i++)
    {
        cudaStreamCreate(&stream[i]);

        size_t index_bytes;
        size_t set_bytes;
        unsigned int current_size;

        if (i == iter_times - 1)
        {
            index_bytes = size_last_iter * sizeof(unsigned int);
            set_bytes = dim * static_cast<size_t>(size_last_iter) * sizeof(float);
            current_size = size_last_iter;
        }
        else
        {
            index_bytes = size_per_iter * sizeof(unsigned int);
            set_bytes = dim * static_cast<size_t>(size_per_iter) * sizeof(float);
            current_size = size_per_iter;
        }

        cudaMalloc((void **)&d_original_sets[i], set_bytes);
        cudaMalloc((void **)&d_indices[i], index_bytes);

        cudaMemcpy(d_original_sets[i], &original_set[i * dim * size_per_iter], set_bytes, cudaMemcpyHostToDevice);

        dim3 block(1024);
        unsigned int grid_dim = ceil(sqrt((current_size + block.x - 1) / block.x));
        dim3 grid(grid_dim, grid_dim);

#ifdef DEBUG
        std::cout << "The " << i << "th iteration of thread " << omp_get_thread_num() << " is workding.\n";
#endif

        belongS2SKernel<<<grid, block, 0, stream[i]>>>(d_indices[i], d_original_sets[i], d_cluster_set, dim, current_size, cluster_size);

        cudaMemcpy(&index[i * size_per_iter], d_indices[i], index_bytes, cudaMemcpyDeviceToHost);

        auto err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            std::cout << "The " << i << "th iteration of thread " << omp_get_thread_num() << ": " << cudaGetErrorString(err) << "\n";
        }
    }

    for (int i = 0; i < iter_times; i++)
    {
        cudaStreamSynchronize(stream[i]);

        cudaFree(d_original_sets[i]);
        cudaFree(d_indices[i]);

        cudaStreamDestroy(stream[i]);
    }
    cudaFree(d_cluster_set);
    free(stream);
    free(d_original_sets);
    free(d_indices);
}

__global__ void kmeansppKernel(unsigned int *indices, float *probability, float *cluster_set, float *cluster_final, unsigned int *omega, int dim, int current_k, int cluster_size)
{
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    extern __shared__ float p_temp[];

    if (idx < cluster_size)
    {
        probability[idx] = INFINITY;

        for (int i = 0; i < current_k; i++)
        {
            p_temp[tid] = 0.0f;
            for (int j = 0; j < dim; j++)
            {
                float diff = cluster_set[idx * dim + j] - cluster_final[i * dim + j];
                p_temp[tid] += diff * diff;
            }
            p_temp[tid] *= omega[idx];

            if (p_temp[tid] < probability[idx])
            {
                probability[idx] = p_temp[tid];
                indices[idx] = i;
            }
        }
    }
}

void cudaKmeanspp(float *cluster_final, float *cluster_set, unsigned int *omega, unsigned int k, const int dim, const unsigned int cluster_size)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, cluster_size - 1);

    unsigned int index = distrib(gen);

    // 均匀分布中随机采样一个原聚类中心集的向量放入最终聚类中心集中
    memcpy(&cluster_final[0], &cluster_set[index * dim], dim * sizeof(float));
    unsigned int current_k = 1;

    float max_p;
    unsigned int max_p_index;

    size_t indices_bytes = cluster_size * sizeof(unsigned int);
    size_t cluster_set_bytes = cluster_size * dim * sizeof(float);
    size_t p_bytes = cluster_size * sizeof(float);

    unsigned int *indices = (unsigned int *)malloc(indices_bytes);
    float *probability = (float *)malloc(p_bytes);

    unsigned int *d_omega, *d_indices;
    float *d_cluster_set, *d_probability;
    cudaMalloc((void **)&d_cluster_set, cluster_set_bytes);
    cudaMalloc((void **)&d_omega, indices_bytes);
    cudaMalloc((void **)&d_indices, indices_bytes);
    cudaMalloc((void **)&d_probability, p_bytes);

    cudaMemcpy(d_cluster_set, cluster_set, cluster_set_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_omega, omega, indices_bytes, cudaMemcpyHostToDevice);

    // 迭代k-1次，每次取一个聚类中心进入c_final
    while (current_k < k)
    {
        max_p = -1.0f;

        cudaStream_t stream;
        cudaStreamCreate(&stream);

        size_t cluster_final_bytes = current_k * dim * sizeof(float);

        float *d_cluster_final;
        cudaMalloc((void **)&d_cluster_final, cluster_final_bytes);

        cudaMemcpy(d_cluster_final, cluster_final, cluster_final_bytes, cudaMemcpyHostToDevice);

        dim3 block(1024);
        dim3 grid((cluster_size + block.x - 1) / block.x);
        kmeansppKernel<<<grid, block, block.x * sizeof(float), stream>>>(d_indices, d_probability, d_cluster_set, d_cluster_final, d_omega, dim, current_k, cluster_size);

        cudaMemcpy(probability, d_probability, p_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(indices, d_indices, indices_bytes, cudaMemcpyDeviceToHost);

        cudaStreamSynchronize(stream);
        cudaFree(d_cluster_final);

#pragma omp parallel for private(i) reduction(max : max_p)
        for (int i = 0; i < cluster_size; i++)
        {
            if (probability[i] > max_p)
            {
                max_p = probability[i];
                max_p_index = indices[i];
            }
        }

        // 将概率最大的向量并入最终聚类中心集
        memcpy(&cluster_final[current_k * dim], &cluster_set[max_p_index * dim], dim * sizeof(float));
        current_k++;
    }

    cudaFree(d_cluster_set);
    cudaFree(d_omega);
    cudaFree(d_indices);
    cudaFree(d_probability);

    free(probability);
    free(indices);
}

__global__ void getNewClusterKernel(float *cluster_new, float *original_set, unsigned int *belong, const int dim,
                                    const unsigned int original_size, unsigned int *count, unsigned int cluster_size)
{
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + tid;

    extern __shared__ float sum[];
    __shared__ unsigned int count_current_blk;
    if (tid < dim)
        sum[tid] = 0.0f;
    if (tid == 0)
        count_current_blk = 0;
    __syncthreads();

    for (unsigned int i = tid, j = 0; j < original_size; i += dim, j++)
    {
        if (belong[j] == bid)
        {
            sum[tid] += original_set[i];
        }
        if (belong[j] == bid && tid == 0)
        {
            count_current_blk++;
        }
    }
    __syncthreads();

    if (tid < dim)
        cluster_new[idx] = sum[tid];

    if (tid == 0)
        count[bid] = count_current_blk;
    __syncthreads();

    if (idx < cluster_size * dim)
        cluster_new[idx] /= count[bid];
}

void cudaGetNewCluster(float *cluster_new, float *original_set, unsigned int *belong, const int dim, const unsigned int original_size, unsigned int cluster_size)
{
    size_t cluster_bytes = dim * cluster_size * sizeof(float);
    size_t origianl_set_bytes = dim * static_cast<size_t>(original_size) * sizeof(float);
    size_t belong_bytes = original_size * sizeof(unsigned int);
    size_t count_bytes = cluster_size * sizeof(unsigned int);

    float *d_cluster_new, *d_original_set;
    unsigned int *d_belong;
    unsigned int *d_count;
    cudaMalloc((void **)&d_cluster_new, cluster_bytes);
    cudaMalloc((void **)&d_original_set, origianl_set_bytes);
    cudaMalloc((void **)&d_belong, belong_bytes);
    cudaMalloc((void **)&d_count, count_bytes);

    cudaMemset(d_cluster_new, 0, cluster_bytes);
    cudaMemcpy(d_original_set, original_set, origianl_set_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_belong, belong, belong_bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_count, 0, count_bytes);

    dim3 block(dim);
    dim3 grid(cluster_size);
    getNewClusterKernel<<<grid, block, (dim) * sizeof(float)>>>(d_cluster_new, d_original_set, d_belong, dim, original_size, d_count, cluster_size);

    cudaMemcpy(cluster_new, d_cluster_new, cluster_bytes, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(d_cluster_new);
    cudaFree(d_original_set);
    cudaFree(d_belong);
    cudaFree(d_count);
}

__global__ void isCloseKernel(float *distance, float *cluster_new, float *cluster_old, float *temp, const int dim, const unsigned int cluster_size)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + tid;

    // 计算每对元素差的平方，存入temp
    if (idx < dim * cluster_size)
    {
        temp[idx] = (cluster_new[idx] - cluster_old[idx]) * (cluster_new[idx] - cluster_old[idx]);
    }
    __syncthreads();

    extern __shared__ float s_data[];
    s_data[tid] = temp[idx];
    __syncthreads();

    // 交错匹配的归约求和
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            s_data[tid] += s_data[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
        distance[blockIdx.x] = s_data[0];
}

bool cudaIsClose(float *cluster_new, float *cluster_old, const int dim, const unsigned int cluster_size, float epsilon)
{
    size_t distance_bytes = cluster_size * sizeof(float);
    size_t cluster_bytes = dim * cluster_size * sizeof(float);

    float *distance = (float *)malloc(distance_bytes);

    float *d_distance, *d_cluster_new, *d_cluster_old, *temp;
    cudaMalloc((void **)&d_distance, distance_bytes);
    cudaMalloc((void **)&d_cluster_new, cluster_bytes);
    cudaMalloc((void **)&d_cluster_old, cluster_bytes);
    cudaMalloc((void **)&temp, cluster_bytes);

    cudaMemcpy(d_cluster_new, cluster_new, cluster_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cluster_old, cluster_old, cluster_bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_distance, 0, distance_bytes);
    cudaMemset(temp, 0, cluster_bytes);

    dim3 block(dim);
    dim3 grid((cluster_size * dim + block.x - 1) / block.x);
    isCloseKernel<<<grid, block, dim * sizeof(float)>>>(d_distance, d_cluster_new, d_cluster_old, temp, dim, cluster_size);

    cudaMemcpy(distance, d_distance, distance_bytes, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(d_distance);
    cudaFree(d_cluster_new);
    cudaFree(d_cluster_old);
    cudaFree(temp);

    for (unsigned int i = 0; i < cluster_size; i++)
    {
        if (distance[i] > epsilon)
        {
            free(distance);
            return false;
        }
    }

    free(distance);
    return true;
}