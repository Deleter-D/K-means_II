#include "../include/vq.h"

void build(float *original_data, unsigned int original_size, int original_dim, char *prefix)
{
    std::string prefix_str = prefix;
    size_t cluster_0_bytes = K0 * original_dim * sizeof(float);
    size_t indices_0_bytes = original_size * sizeof(unsigned int);

    float *cluster_0 = (float *)malloc(cluster_0_bytes);
    randomKmeans(original_data, original_size, original_dim, cluster_0, K0);

    unsigned int *indices_0 = (unsigned int *)malloc(indices_0_bytes);
    cudaBelongS2S(indices_0, original_data, cluster_0, original_dim, original_size, K0);

    save(cluster_0, cluster_0_bytes, prefix_str + "cluster_0");
    save(indices_0, indices_0_bytes, prefix_str + "indices_0");

    size_t cluster_1_bytes = K1 * original_dim * sizeof(float);
    size_t indices_1_bytes = K0 * sizeof(unsigned int);

    float *cluster_1 = (float *)malloc(cluster_1_bytes);
    randomKmeans(cluster_0, K0, original_dim, cluster_1, K1);

    unsigned int *indices_1 = (unsigned int *)malloc(indices_1_bytes);
    cudaBelongS2S(indices_1, cluster_0, cluster_1, original_dim, K0, K1);

    save(cluster_1, cluster_1_bytes, prefix_str + "cluster_1");
    save(indices_1, indices_1_bytes, prefix_str + "indices_1");

    free(cluster_0);
    free(indices_0);
    free(cluster_1);
    free(indices_1);
}

__global__ void levelQueryKernel(unsigned int *result, float *query_set, float *cluster, float *indices, unsigned int *belong, unsigned int dim, unsigned int query_size, unsigned int cluster_size, unsigned int indices_size)
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

void query(float *query_set, unsigned int query_size, float *original_data, unsigned int original_dim, unsigned int original_size, unsigned int topk, float *cluster_0, float *cluster_1, unsigned int *indices_0, unsigned int *indices_1)
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

    // 第一层查询
    size_t belong_bytes = query_size * sizeof(unsigned int);
    // unsigned int *belong_1 = (unsigned int *)malloc(belong_bytes);
    unsigned int *d_belong_1;
    cudaMalloc((void **)&d_belong_1, belong_bytes);

    dim3 block(1024);
    dim3 grid((query_size + block.x - 1) / block.x);

    belongS2SKernel<<<grid, block>>>(d_belong_1, d_query_set, d_cluster_1, original_dim, query_size, K1);

    // cudaMemcpy(belong_1, d_belong_1, belong_bytes, cudaMemcpyDeviceToHost);

    cudaStreamSynchronize(stream);
    // cudaFree(d_belong_1);

    // 第二层查询

    unsigned int *d_belong_0;
    cudaMalloc((void **)&d_belong_0, belong_bytes);

    levelQueryKernel<<<grid, block>>>(d_belong_0)
}