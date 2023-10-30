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

    float *x, *S;
    x = (float *)malloc(DIM * sizeof(float));
    S = (float *)malloc(SIZE * DIM * sizeof(float));
    for (int i = 0; i < SIZE * DIM; i++)
    {
        S[i] = distrib(gen);
    }
    for (int i = 0; i < DIM; i++)
    {
        x[i] = distrib(gen);
    }

    float cost = op.costFromV2S(x, S, DIM, SIZE);

    float *y = (float *)malloc(DIM * sizeof(float));
    memcpy(y, &S[0], DIM * sizeof(float));

    float distance = op.euclideanDistance(x, y, DIM);

    if (cost > distance)
        return -1;

    return 0;
}
