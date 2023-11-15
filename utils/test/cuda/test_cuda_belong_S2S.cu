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

    size_t *index = (size_t *)malloc(TEST_SIZE * sizeof(size_t));
    belongS2S(index, S1, S2, TEST_DIM, TEST_SIZE, TEST_SIZE);

    std::cout << "host finished." << std::endl;

    size_t *index_cuda = (size_t *)malloc(TEST_SIZE * sizeof(size_t));
    cudaBelongS2S(index_cuda, S1, S2, TEST_DIM, TEST_SIZE, TEST_SIZE);

    free(S1);
    free(S2);

    for (int i = 0; i < TEST_SIZE; i++)
    {
        if (index[i] != index_cuda[i])
        {
            std::cout << "host: " << index[i] << ", cuda: " << index_cuda[i] << "\n";
            // free(index);
            // free(index_cuda);
            // return -1;
        }
    }

    free(index);
    free(index_cuda);

    return 0;
}
