#include <iostream>
#include <random>
#include <cstring>
#include "../include/pq.h"

int main(int argc, char const *argv[])
{
    cudaSetDevice(0);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> uniform(0, K - 1);
    std::uniform_real_distribution<float> distrib(0, 1);

    float *distance, *distance_cuda, *distance_tab;
    size_t *index;
    distance = (float *)malloc(TEST_SIZE * sizeof(float));
    distance_cuda = (float *)malloc(TEST_SIZE * sizeof(float));
    distance_tab = (float *)malloc(K * sizeof(float));
    index = (size_t *)malloc(TEST_SIZE * sizeof(size_t));

    for (int i = 0; i < TEST_SIZE; i++)
    {
        index[i] = uniform(gen);
    }
    for (int i = 0; i < K; i++)
    {
        distance_tab[i] = distrib(gen);
    }

    cudaGetAsymmetricDistance(distance_cuda, distance_tab, index, TEST_SIZE);

    for (size_t i = 0; i < TEST_SIZE; i++)
    {
        distance[i] = distance_tab[index[i]];
    }

    free(distance_tab);
    free(index);

    for (int i = 0; i < TEST_SIZE; i++)
    {
        if (distance[i] != distance_cuda[i])
        {
            // printf("%d: %f, %f\n", i, distance[i], distance_cuda[i]);
            free(distance);
            free(distance_cuda);
            return -1;
        }
    }

    free(distance);
    free(distance_cuda);

    cudaDeviceReset();

    return 0;
}
