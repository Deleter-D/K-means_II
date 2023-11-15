#include <iostream>
#include <random>
#include <cstring>
#include "../../include/common.cuh"
#include "../../include/common.h"
#include "../../include/config.h"

int main(int argc, char const *argv[])
{
    cudaSetDevice(0);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> distrib(0, 1);

    float *vec, *set;
    vec = (float *)malloc(TEST_DIM * sizeof(float));
    set = (float *)malloc(TEST_SIZE * TEST_DIM * sizeof(float));

    for (int i = 0; i < TEST_DIM; i++)
    {
        vec[i] = distrib(gen);
    }
    for (int i = 0; i < TEST_SIZE * TEST_DIM; i++)
    {
        set[i] = distrib(gen);
    }
    float *distance_cuda = (float *)malloc(TEST_SIZE * sizeof(float));
    cudaEuclideanDistance(distance_cuda, vec, set, TEST_DIM, TEST_SIZE);

    float temp;
    for (int i = 0; i < TEST_SIZE; i++)
    {
        temp = euclideanDistance(vec, &set[i * TEST_DIM], TEST_DIM);
        if (abs(distance_cuda[i] - temp) > 1e-5)
        {
            printf("%d: %f,%f\n", i, temp, distance_cuda[i]);
            free(vec);
            free(set);
            free(distance_cuda);
            return -1;
        }
    }

    free(vec);
    free(set);
    free(distance_cuda);

    return 0;
}
