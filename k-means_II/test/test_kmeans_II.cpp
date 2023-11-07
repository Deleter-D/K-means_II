#include <iostream>
#include <random>
#include <cstring>
#include <cuda_runtime.h>
#include "../include/kmeans_II.h"
#include "../../utils/include/config.h"

int main(int argc, char const *argv[])
{
    std::random_device rd;
    std::mt19937 gen(rd());
    // std::uniform_real_distribution<float> distrib(0, 1);
    float st = 30.f;
    std::normal_distribution<float> distrib(0, st);

    float *INPUT, *cluster_set;
    INPUT = (float *)malloc(TEST_SIZE * TEST_DIM * sizeof(float));
    cluster_set = (float *)malloc(K * TEST_DIM * sizeof(float));
    for (int i = 0; i < TEST_SIZE * TEST_DIM; i++)
    {
        INPUT[i] = distrib(gen);
    }

    init(INPUT, TEST_SIZE, TEST_DIM, cluster_set);
    iteration(INPUT, TEST_SIZE, TEST_DIM, cluster_set);

    free(INPUT);
    free(cluster_set);

    return 0;
}
