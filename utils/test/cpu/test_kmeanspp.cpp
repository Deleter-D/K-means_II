#include <iostream>
#include <random>
#include <cstring>
#include "../../include/common.h"
#include "../../include/config.h"

int main(int argc, char const *argv[])
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> distrib(0, 1);

    float *cluster_set;
    unsigned int *omega;
    omega = (unsigned int *)malloc(TEST_SIZE * sizeof(unsigned int));
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

    free(omega);
    free(cluster_set);

    for (int i = 0; i < K * TEST_DIM; i++)
    {
        if (cluster_final[i] < 0 || cluster_final[i] > 1)
        {
            free(cluster_final);
            return -1;
        }
    }

    free(cluster_final);

    return 0;
}