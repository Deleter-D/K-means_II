#include <iostream>
#include "../include/pq_util.cuh"

__global__ void getAsymmetricDistanceKernel(float *distance, float *distance_tab, unsigned int *index, unsigned int size)
{
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.y * gridDim.x + blockIdx.x;
    unsigned int idx = bid * blockDim.x + tid;

    if (idx < size)
    {
        distance[idx] = distance_tab[index[idx]];
    }
}

void cudaGetAsymmetricDistance(float *distance, float *distance_tab, unsigned int *index, const unsigned int size)
{
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    size_t distance_bytes = size * sizeof(float);
    size_t tab_bytes = K * sizeof(float);
    size_t index_bytes = size * sizeof(unsigned int);

    float *d_distance, *d_distance_tab;
    unsigned int *d_index;
    cudaMalloc((void **)&d_distance, distance_bytes);
    cudaMalloc((void **)&d_distance_tab, tab_bytes);
    cudaMalloc((void **)&d_index, index_bytes);

    cudaMemcpy(d_distance, distance, distance_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_distance_tab, distance_tab, tab_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_index, index, index_bytes, cudaMemcpyHostToDevice);

    dim3 block(1024);
    unsigned int grid_dim = ceil(sqrt((size + block.x - 1) / block.x));
    dim3 grid(grid_dim, grid_dim);
    getAsymmetricDistanceKernel<<<grid, block>>>(d_distance, d_distance_tab, d_index, size);

    cudaMemcpy(distance, d_distance, distance_bytes, cudaMemcpyDeviceToHost);

    cudaStreamSynchronize(stream);

    cudaFree(d_distance);
    cudaFree(d_distance_tab);
    cudaFree(d_index);
}