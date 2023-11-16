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

    float *x, *S;
    x = (float *)malloc(TEST_DIM * sizeof(float));
    S = (float *)malloc(TEST_SIZE * TEST_DIM * sizeof(float));
    for (int i = 0; i < TEST_SIZE * TEST_DIM; i++)
    {
        S[i] = distrib(gen);
    }
    for (int i = 0; i < TEST_DIM; i++)
    {
        x[i] = distrib(gen);
    }

    unsigned int index = belongV2S(x, S, TEST_DIM, TEST_SIZE);

    free(x);
    free(S);

    if (index < 0 || index >= TEST_SIZE)
        return -1;

    return 0;
}
