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
    float *distance_cuda = cudaEuclideanDistance(vec, set, TEST_DIM, TEST_SIZE);
    for (int i = 0; i < TEST_SIZE; i++)
    {
        float temp = euclideanDistance(vec, &set[i * TEST_DIM], TEST_DIM);
        if (abs(distance_cuda[i] - temp) > 1e-5)
            return -1;
    }

    return 0;
}
