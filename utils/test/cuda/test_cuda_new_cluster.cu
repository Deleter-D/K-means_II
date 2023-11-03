#include <iostream>
#include <random>
#include <cstring>
#include "../../include/common.h"
#include "../../include/common.cuh"
#include "../../include/config.h"

int main(int argc, char const *argv[])
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> distrib(0, 50000000);
    std::uniform_int_distribution<> distribInt(0, K - 1);

    float *original;
    size_t *belong;
    original = (float *)malloc(TEST_SIZE * TEST_DIM * sizeof(float));
    belong = (size_t *)malloc(TEST_SIZE * sizeof(size_t));
    for (int i = 0; i < TEST_SIZE * TEST_DIM; i++)
    {
        original[i] = distrib(gen);
    }
    for (int i = 0; i < TEST_SIZE; i++)
    {
        belong[i] = distribInt(gen);
    }

    float *cluster_new = (float *)malloc(TEST_DIM * K * sizeof(float));
    float *mean_vec = (float *)malloc(TEST_DIM * sizeof(float));
    memset(mean_vec, 0, TEST_DIM * sizeof(float));
    for (int i = 0; i < K; i++)
    {
        memset(mean_vec, 0, TEST_DIM * sizeof(float));
        meanVec(mean_vec, original, belong, TEST_DIM, TEST_SIZE, i);
        memcpy(&cluster_new[i * TEST_DIM], mean_vec, TEST_DIM * sizeof(float));
    }
    free(mean_vec);

    float *cluster_new_cuda = (float *)malloc(TEST_DIM * K * sizeof(float));
    cudaGetNewCluster(cluster_new_cuda, original, belong, TEST_DIM, TEST_SIZE);

    free(original);
    free(belong);

    for (int i = 0; i < K; i++)
    {
        for (int j = 0; j < TEST_DIM; j++)
        {
            if (abs(cluster_new_cuda[i * TEST_DIM + j] - cluster_new[i * TEST_DIM + j]) > 1e-8)
            {
                printf("%d,%d: %f, %f\n", i, j, cluster_new[i * TEST_DIM + j], cluster_new_cuda[i * TEST_DIM + j]);
                free(cluster_new);
                free(cluster_new_cuda);
                return -1;
            }
        }
    }

    free(cluster_new);
    free(cluster_new_cuda);

    return 0;
}
