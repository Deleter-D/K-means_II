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
    std::uniform_int_distribution<> distribInt(0, K);

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
    }

    float *mean = meanOfV(original, belong, TEST_DIM, TEST_SIZE, 4);

    for (int i = 0; i < TEST_DIM; i++)
    {
        if (mean[i] < 0 || mean[i] >= 1)
            return -1;
    }

    return 0;
}
