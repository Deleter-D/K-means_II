#include "../include/common.cuh"
#include "../include/config.h"
#include <stdio.h>
#include <random>
#include <omp.h>

__global__ void euclideanDistanceKernel(float *distance, float *vec, float *set, float *temp, const int dim, const int size)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + tid;

    // 计算每对元素差的平方，存入temp
    if (idx < dim * size)
    {
        temp[idx] = (vec[tid] - set[idx]) * (vec[tid] - set[idx]);
    }
    __syncthreads();

    extern __shared__ float s_data[];
    s_data[tid] = temp[idx];
    // if (tid < 32)
    //     s_data[tid] = temp[idx] + temp[idx + 32] + temp[idx + 64];
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

void cudaEuclideanDistance(float *distance, float *vec, float *set, const int dim, const int size)
{
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    size_t distance_bytes = size * sizeof(float);
    size_t vec_bytes = dim * sizeof(float);
    size_t set_bytes = dim * size * sizeof(float);

    float *d_distance, *d_vec, *d_set, *temp;
    cudaMalloc((void **)&d_distance, distance_bytes);
    cudaMalloc((void **)&d_vec, vec_bytes);
    cudaMalloc((void **)&d_set, set_bytes);
    cudaMalloc((void **)&temp, set_bytes);

    cudaMemcpy(d_vec, vec, vec_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_set, set, set_bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_distance, 0, distance_bytes);
    cudaMemset(temp, 0, set_bytes);

    dim3 block(dim);
    dim3 grid((size * dim + block.x - 1) / block.x);
    euclideanDistanceKernel<<<grid, block, dim * sizeof(float), stream>>>(d_distance, d_vec, d_set, temp, dim, size);

    cudaMemcpy(distance, d_distance, distance_bytes, cudaMemcpyDeviceToHost);
    cudaStreamSynchronize(stream);

    cudaFree(d_distance);
    cudaFree(d_vec);
    cudaFree(d_set);
    cudaFree(temp);

    cudaStreamDestroy(stream);
}

float cudaCostFromV2S(float *vec, float *cluster_set, const int dim, const size_t size)
{
    float min = 3.40282347e+38F;

    float *distance = (float *)malloc(size * sizeof(float));
    cudaEuclideanDistance(distance, vec, cluster_set, dim, size);

    // TODO: 取最小值可优化
    for (size_t i = 0; i < size; i++)
    {
        if (distance[i] < min)
            min = distance[i];
    }

    free(distance);

    return min;
}

float cudaCostFromS2S(float *original_set, float *cluster_set, const int dim, const size_t original_size, const size_t cluster_size)
{
    float *sums = (float *)malloc(original_size * sizeof(float));
#pragma omp parallel for
    for (size_t i = 0; i < original_size; i++)
    {
        sums[i] = cudaCostFromV2S(&original_set[i * dim], cluster_set, dim, cluster_size);
    }
    cudaDeviceSynchronize();

    float sum = 0.0f;
#pragma omp parallel for reduction(+ : sum)
    for (size_t i = 0; i < original_size; i++)
    {
        sum += sums[i];
    }

    free(sums);

    return sum;
}

size_t cudaBelongV2S(float *x, float *cluster_set, const int dim, const size_t size)
{
    float min = 3.40282347e+38F;
    size_t index;

    float *distance = (float *)malloc(size * sizeof(float));
    cudaEuclideanDistance(distance, x, cluster_set, dim, size);

    // TODO: 取最小值可优化
    for (size_t i = 0; i < size; i++)
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

void cudaBelongS2S(size_t *index, float *original_set, float *cluster_set, const int dim, const size_t original_size, const size_t cluster_size)
{
#pragma omp parallel for
    for (size_t i = 0; i < original_size; i++)
    {
        index[i] = cudaBelongV2S(&original_set[i * dim], cluster_set, dim, cluster_size);
    }
    cudaDeviceSynchronize();
}

void cudaKmeanspp(float *cluster_final, float *cluster_set, size_t *omega, size_t k, const int dim, const size_t cluster_size)
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
        float cost_set2final = cudaCostFromS2S(cluster_set, cluster_final, dim, cluster_size, current_k);
#pragma omp parallel for
        for (size_t i = 0; i < cluster_size; i++)
        {
            // 计算当前向量的概率
            temp_p = omega[i] * cudaCostFromV2S(&cluster_set[i * dim], cluster_final, dim, current_k) / cost_set2final;
#pragma omp critical
            {
                // 记录概率最大的向量信息
                if (temp_p > max_p)
                {
                    max_p = temp_p;
                    max_p_index = i;
                }
            }
        }
        // 将概率最大的向量并入最终聚类中心集
        memcpy(&cluster_final[current_k * dim], &cluster_set[max_p_index * dim], dim * sizeof(float));
        current_k++;
    }
}

__global__ void getNewClusterKernel(float *cluster_new, float *original_set, size_t *belong, const int dim, const size_t original_size, unsigned int *count)
{
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + tid;

    extern __shared__ float sum[];
    __shared__ size_t count_current_blk;
    if (tid < dim)
        sum[tid] = 0.0f;
    if (tid == 0)
        count_current_blk = 0;
    __syncthreads();

    for (size_t i = tid, j = 0; j < original_size; i += dim, j++)
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

    if (idx < K * dim)
        cluster_new[idx] /= count[bid];
}

void cudaGetNewCluster(float *cluster_new, float *original_set, size_t *belong, const int dim, const size_t original_size)
{
    size_t cluster_bytes = dim * K * sizeof(float);
    size_t origianl_set_bytes = dim * original_size * sizeof(float);
    size_t belong_bytes = original_size * sizeof(size_t);
    size_t count_bytes = K * sizeof(unsigned int);

    float *d_cluster_new, *d_original_set;
    size_t *d_belong;
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
    dim3 grid(K);
    getNewClusterKernel<<<grid, block, (dim) * sizeof(float)>>>(d_cluster_new, d_original_set, d_belong, dim, original_size, d_count);

    cudaMemcpy(cluster_new, d_cluster_new, cluster_bytes, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(d_cluster_new);
    cudaFree(d_original_set);
    cudaFree(d_belong);
    cudaFree(d_count);
}

__global__ void isCloseKernel(float *distance, float *cluster_new, float *cluster_old, float *temp, const int dim, const size_t cluster_size)
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

bool cudaIsClose(float *cluster_new, float *cluster_old, const int dim, const size_t cluster_size, float epsilon)
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

    for (size_t i = 0; i < cluster_size; i++)
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