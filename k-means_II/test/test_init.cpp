#include <iostream>
#include <random>
#include <cstring>
#include "../include/kmeans.h"

#define DIM 96
#define SIZE 300
#define K 10

int main(int argc, char const *argv[])
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> distrib(0, 1);

    float *INPUT;
    INPUT = (float *)malloc(SIZE * DIM * sizeof(float));
    for (int i = 0; i < SIZE * DIM; i++)
    {
        INPUT[i] = distrib(gen);
    }

    kmeans op(INPUT, SIZE, DIM, K);

    op.init();

    for (int i = 0; i < (2 * K * 5 + 2) * DIM; i++)
    {
        if (op.center[i] < 0 && op.center[i] >= 1)
            return -1;
    }

    return 0;
}
