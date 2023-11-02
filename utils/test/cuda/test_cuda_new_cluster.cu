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
    std::uniform_real_distribution<float> distrib(0, 1);
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
    for (int i = 0; i < K; i++)
    {
        memcpy(&cluster_new[i * TEST_DIM], meanVec(original, belong, TEST_DIM, TEST_SIZE, i), TEST_DIM * sizeof(float));
    }

    float *cluster_new_cuda = cudaGetNewCluster(original, belong, TEST_DIM, TEST_SIZE);

    for (int i = 0; i < K; i++)
    {
        for (int j = 0; j < TEST_DIM; j++)
        {
            if (abs(cluster_new_cuda[i * TEST_DIM + j] - cluster_new[i * TEST_DIM + j]) > 5e-2)
            {
                // printf("%d,%d: %f, %f\n", i, j, cluster_new[i * TEST_DIM + j], cluster_new_cuda[i * TEST_DIM + j]);
                return -1;
            }
        }
    }

    return 0;
}
