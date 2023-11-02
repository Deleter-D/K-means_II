#include <iostream>
#include <random>
#include <cstring>
#include "../include/kmeans_II.h"
#include "../../utils/include/config.h"

#define __USE_CUDA__

int main(int argc, char const *argv[])
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> distrib(0, 1);

    float *INPUT;
    INPUT = (float *)malloc(TEST_SIZE * TEST_DIM * sizeof(float));
    for (int i = 0; i < TEST_SIZE * TEST_DIM; i++)
    {
        INPUT[i] = distrib(gen);
    }

    kmeans_II op(INPUT, TEST_SIZE, TEST_DIM, K);

    op.init();
    op.iteration();

    for (int i = 0; i < K * TEST_DIM; i++)
    {
        if (op.cluster_set[i] < 0 || op.cluster_set[i] >= 1)
            return -1;
    }

    return 0;
}
