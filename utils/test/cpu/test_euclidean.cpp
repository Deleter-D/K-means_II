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
        y[i] = distrib(gen);
    }
    float distance1 = euclideanDistance(x, x, TEST_DIM);
    float distance2 = euclideanDistance(x, y, TEST_DIM);

    if (distance1 != 0)
        return -1;

    if (distance2 == 0)
        return -1;

    return 0;
}
