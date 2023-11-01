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
        printf("----->%d\n", belong[i]);
    }

    float *cluster_new = cudaGetNewCluster(original, belong, TEST_DIM, TEST_SIZE);

    for (int i = 0; i < K; i++)
    {
        float *temp = meanVec(original, belong, TEST_DIM, TEST_SIZE, i);
        for (int j = 0; j < TEST_DIM; j++)
        {
            if (abs(cluster_new[i * TEST_DIM + j] - temp[j]) > 1e-5)
            {
                // return -1;
                // printf("%d,%d: %f, %f\n", i, j, temp[j], cluster_new[i * TEST_DIM + j]);
            }
        }
    }

    return 0;
}