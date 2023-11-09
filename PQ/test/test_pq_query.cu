#include <random>
#include <cstring>
#include <omp.h>
#include <unistd.h>
#include "../include/pq.h"

int main(int argc, char const *argv[])
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> distrib(0, 1);
    std::normal_distribution<float> normal(5, 10);

    float *S1;
    size_t s1_bytes = size_t(TEST_SIZE) * TEST_TOTAL_DIM * sizeof(float);
    S1 = (float *)malloc(s1_bytes);

    // #pragma omp parallel for
    for (size_t i = 0; i < size_t(TEST_SIZE) * TEST_TOTAL_DIM; i++)
    {
        S1[i] = normal(gen);
    }

    std::string filename("original_data");
    save(S1, size_t(TEST_SIZE) * TEST_TOTAL_DIM, filename);

    productQuantizationBuild(filename, TEST_SIZE, TEST_TOTAL_DIM, M);

    float *input = (float *)malloc(10 * TEST_TOTAL_DIM * sizeof(float));
    size_t *result = (size_t *)malloc(10 * TOPK * sizeof(size_t));
    // #pragma omp parallel for
    for (int i = 0; i < 10 * TEST_TOTAL_DIM; i++)
    {
        input[i] = normal(gen);
    }
    memset(result, 0, 10 * TOPK * sizeof(size_t));

    productQuantizationQuery(result, input, 10, TEST_SIZE, TEST_TOTAL_DIM, M, TOPK);

    for (int i = 0; i < 10 * TOPK; i++)
    {
        printf("%ld\t", result[i]);
        if (i != 0 && (i + 1) % TOPK == 0)
        {
            printf("\n");
        }
    }

    free(S1);
    free(input);
    free(result);

    return 0;
}