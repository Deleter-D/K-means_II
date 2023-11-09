#include <random>
#include <cstring>
#include <omp.h>
#include "../include/pq.h"

int main(int argc, char const *argv[])
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> distrib(0, 1);

    float *S1;
    size_t s1_bytes = size_t(TEST_SIZE) * TEST_TOTAL_DIM * sizeof(float);
    S1 = (float *)malloc(s1_bytes);

#pragma omp parallel for
    for (size_t i = 0; i < size_t(TEST_SIZE) * TEST_TOTAL_DIM; i++)
    {
        S1[i] = distrib(gen);
    }

    std::string filename("original_data");
    save(S1, size_t(TEST_SIZE) * TEST_TOTAL_DIM, filename);

    productQuantizationBuild(filename, TEST_SIZE, TEST_TOTAL_DIM, M);

    free(S1);

    return 0;
}