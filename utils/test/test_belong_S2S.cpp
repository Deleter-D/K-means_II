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

    float *S1, *S2;
    S1 = (float *)malloc(TEST_SIZE * TEST_DIM * sizeof(float));
    S2 = (float *)malloc(TEST_SIZE * TEST_DIM * sizeof(float));
    for (int i = 0; i < TEST_SIZE * TEST_DIM; i++)
    {
        S1[i] = distrib(gen);
        S2[i] = distrib(gen);
    }

    size_t *index = belongS2S(S1, S2, TEST_DIM, TEST_SIZE, TEST_SIZE);

    for (int i = 0; i < TEST_SIZE; i++)
    {
        if (index[i] < 0 || index[i] >= TEST_SIZE)
            return -1;
    }

    return 0;
}
