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

    float *cluster_set;
    size_t *omega;
    omega = (size_t *)malloc(TEST_SIZE * sizeof(size_t));
    cluster_set = (float *)malloc(TEST_SIZE * TEST_DIM * sizeof(float));
    for (int i = 0; i < TEST_SIZE * TEST_DIM; i++)
    {
        cluster_set[i] = distrib(gen);
    }
    for (int i = 0; i < TEST_SIZE; i++)
    {
        omega[i] = distrib(gen);
    }

    float *cluster_final = (float *)malloc(K * TEST_DIM * sizeof(float));
    kmeanspp(cluster_final, cluster_set, omega, K, TEST_DIM, TEST_SIZE);
    float *cluster_final_cuda = (float *)malloc(K * TEST_DIM * sizeof(float));
    cudaKmeanspp(cluster_final_cuda, cluster_set, omega, K, TEST_DIM, TEST_SIZE);

    free(omega);
    free(cluster_set);

    for (int i = TEST_DIM; i < K * TEST_DIM; i++)
    {
        if (cluster_final[i] != cluster_final_cuda[i])
        {
            free(cluster_final);
            free(cluster_final_cuda);
            return -1;
        }
    }

    free(cluster_final);
    free(cluster_final_cuda);

    return 0;
}