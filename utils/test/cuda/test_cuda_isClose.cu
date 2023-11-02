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

    float *S1, *S2;
    S1 = (float *)malloc(TEST_SIZE * TEST_DIM * sizeof(float));
    S2 = (float *)malloc(TEST_SIZE * TEST_DIM * sizeof(float));
    for (int i = 0; i < TEST_SIZE * TEST_DIM; i++)
    {
        S1[i] = distrib(gen);
        S2[i] = distrib(gen);
    }

    bool isclose = isClose(S1, S2, TEST_DIM, TEST_SIZE, THRESHOLD);
    bool isclose_cuda = cudaIsClose(S1, S2, TEST_DIM, TEST_SIZE, THRESHOLD);

    if (isclose != isclose_cuda)
        return -1;

    return 0;
}
