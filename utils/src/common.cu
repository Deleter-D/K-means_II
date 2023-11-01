#include "../include/common.cuh"
#include <stdio.h>

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

    // 将每个向量的96个元素归约为32个元素
    __shared__ float s_data[32];
    if (tid < 32)
        s_data[tid] = temp[idx] + temp[idx + 32] + temp[idx + 64];
    __syncthreads();

    // 交错匹配的归约求和
    for (int stride = 16; stride > 0; stride >>= 1)
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

float *cudaEuclideanDistance(float *vec, float *set, const int dim, const int size)
{
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    size_t distance_bytes = size * sizeof(float);
    size_t vec_bytes = dim * sizeof(float);
    size_t set_bytes = dim * size * sizeof(float);

    float *distance = (float *)malloc(distance_bytes);

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
    euclideanDistanceKernel<<<grid, block, 32 * sizeof(float), stream>>>(d_distance, d_vec, d_set, temp, dim, size);

    cudaMemcpy(distance, d_distance, distance_bytes, cudaMemcpyDeviceToHost);
    cudaStreamSynchronize(stream);

    cudaStreamDestroy(stream);

    return distance;
}

float cudaCostFromV2S(float *vec, float *cluster_set, const int dim, const size_t size)
{
    float min = MAXFLOAT;

    float *distance = cudaEuclideanDistance(vec, cluster_set, dim, size);

    // TODO: 取最小值可优化
    for (size_t i = 0; i < size; i++)
    {
        if (distance[i] < min)
            min = distance[i];
    }

    return min;
}

float cudaCostFromS2S(float *original_set, float *cluster_set, const int dim, const size_t original_size, const size_t cluster_size)
{
    float *sums = (float *)malloc(original_size * sizeof(float));
    for (size_t i = 0; i < original_size; i++)
    {
        sums[i] = cudaCostFromV2S(&original_set[i * dim], cluster_set, dim, cluster_size);
    }
    cudaDeviceSynchronize();

    float sum = 0.0f;
    for (size_t i = 0; i < original_size; i++)
    {
        printf("%f\n", sums[i]);
        sum += sums[i];
    }
    return sum;
}

size_t cudaBelongV2S(float *x, float *cluster_set, const int dim, const size_t size)
{
    float min = MAXFLOAT;
    size_t index;

    float *distance = cudaEuclideanDistance(x, cluster_set, dim, size);

    // TODO: 取最小值可优化
    for (size_t i = 0; i < size; i++)
    {
        if (distance[i] < min)
        {
            min = distance[i];
            index = i;
        }
    }

    return index;
}