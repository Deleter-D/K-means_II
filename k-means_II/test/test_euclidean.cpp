#include <iostream>
#include <random>
#include <cstring>
#include "../include/kmeans.h"

#define DIM 96
#define SIZE 100
#define K 10

int main(int argc, char const *argv[])
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> distrib(0, 1);

    kmeans op;

    float *x, *y;
    x = (float *)malloc(DIM * sizeof(float));
    y = (float *)malloc(DIM * sizeof(float));

    for (int i = 0; i < DIM; i++)
    {
        x[i] = distrib(gen);
        y[i] = distrib(gen);
    }
    float distance1 = op.euclideanDistance(x, x, DIM);
    float distance2 = op.euclideanDistance(x, y, DIM);

    if (distance1 != 0)
        return -1;

    if (distance2 == 0)
        return -1;

    return 0;
}
