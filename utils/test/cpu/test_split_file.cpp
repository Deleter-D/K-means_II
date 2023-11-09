#include <random>
#include <cstring>
#include <omp.h>
#include <sys/types.h>
#include "../../include/common.h"
#include "../../include/config.h"

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
    save(S1, TEST_SIZE * TEST_TOTAL_DIM, filename);

    split_file(filename, TEST_SIZE, TEST_TOTAL_DIM, M);

    float **subsets = (float **)malloc(M * sizeof(float *));
    for (unsigned int i = 0; i < M; i++)
    {
        subsets[i] = (float *)malloc(TEST_SIZE * TEST_DIM * sizeof(float));
        load(subsets[i], TEST_SIZE * TEST_DIM, "subset" + std::to_string(i));
    }

    for (unsigned int i = 0; i < M; i++)
    {
        for (size_t j = 0; j < TEST_SIZE; j++)
        {
            for (int k = 0; k < TEST_DIM; k++)
            {
                if (S1[j * TEST_TOTAL_DIM + i * TEST_DIM + k] != subsets[i][j * TEST_DIM + k])
                {
                    printf("(%d, %ld, %d): %f, %f\n", i, j, k, S1[j * TEST_TOTAL_DIM + i * TEST_DIM + k], subsets[i][j * TEST_DIM + k]);
                }
            }
        }
    }

    for (unsigned int i = 0; i < M; i++)
    {
        free(subsets[i]);
    }
    free(subsets);
    free(S1);

    return 0;
}