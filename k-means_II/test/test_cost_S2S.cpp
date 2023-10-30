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

    float *S1, *S2;
    S1 = (float *)malloc(SIZE * DIM * sizeof(float));
    S2 = (float *)malloc(SIZE * DIM * sizeof(float));
    for (int i = 0; i < SIZE * DIM; i++)
    {
        S1[i] = distrib(gen);
        S2[i] = distrib(gen);
    }

    float cost = op.costFromS2S(S1, S2, DIM, SIZE, SIZE);

    return 0;
}
