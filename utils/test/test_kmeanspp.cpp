#include <iostream>
#include <random>
#include <cstring>
#include "../include/common.h"
#include "../include/config.h"

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

    float *cluster_final = kmeanspp(cluster_set, omega, K, TEST_DIM, TEST_SIZE);

    for (int i = 0; i < K * TEST_DIM; i++)
    {
        if (cluster_final[i] < 0 || cluster_final[i] > 1)
            return -1;
    }

    return 0;
}