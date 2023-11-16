#include <iostream>
#include <random>
#include <cstring>
#include "../include/kmeans_II.h"
#include "../../utils/include/config.h"

int main(int argc, char const *argv[])
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> distrib(0, 1);

    float *INPUT, *cluster_set;
    INPUT = (float *)malloc(TEST_SIZE * TEST_DIM * sizeof(float));
    cluster_set = (float *)malloc(K * TEST_DIM * sizeof(float));
    for (int i = 0; i < TEST_SIZE * TEST_DIM; i++)
    {
        INPUT[i] = distrib(gen);
    }

    init(INPUT, TEST_SIZE, TEST_DIM, cluster_set);

    free(INPUT);

    for (int i = 0; i < K * TEST_DIM; i++)
    {
        if (cluster_set[i] < 0 || cluster_set[i] >= 1)
        {
            printf("%d: %f\n", i, cluster_set[i]);
            free(cluster_set);
            return -1;
        }
    }

    free(cluster_set);
    return 0;
}
