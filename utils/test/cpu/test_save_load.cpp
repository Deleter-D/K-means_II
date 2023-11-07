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

    float *x, *y;
    x = (float *)malloc(TEST_DIM * sizeof(float));
    y = (float *)malloc(TEST_DIM * sizeof(float));

    for (int i = 0; i < TEST_DIM; i++)
    {
        x[i] = distrib(gen);
    }
    memset(y, 0, TEST_DIM);

    save(x, TEST_DIM, "data");
    load(y, TEST_DIM, "data");

    for (int i = 0; i < TEST_DIM; i++)
    {
        if (x[i] != y[i])
            return -1;
    }

    size_t *x1, *y1;
    x1 = (size_t *)malloc(TEST_DIM * sizeof(size_t));
    y1 = (size_t *)malloc(TEST_DIM * sizeof(size_t));

    for (int i = 0; i < TEST_DIM; i++)
    {
        x1[i] = distrib(gen);
    }
    memset(y1, 0, TEST_DIM);

    save(x1, TEST_DIM, "data1");
    load(y1, TEST_DIM, "data1");

    for (int i = 0; i < TEST_DIM; i++)
    {
        if (x1[i] != y1[i])
            return -1;
    }

    free(x);
    free(y);
    free(x1);
    free(y1);

    return 0;
}