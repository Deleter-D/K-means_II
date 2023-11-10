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

#pragma omp parallel for
    for (size_t i = 0; i < size_t(TEST_SIZE) * TEST_TOTAL_DIM; i++)
    {
        S1[i] = normal(gen);
    }

    productQuantizationBuild(S1, TEST_SIZE, TEST_TOTAL_DIM, M, "./output/");

    free(S1);

    size_t input_size = 10;
    float *input = (float *)malloc(input_size * TEST_TOTAL_DIM * sizeof(float));
    size_t *result = (size_t *)malloc(input_size * TOPK * sizeof(size_t));
#pragma omp parallel for
    for (int i = 0; i < input_size * TEST_TOTAL_DIM; i++)
    {
        input[i] = normal(gen);
    }
    memset(result, 0, input_size * TOPK * sizeof(size_t));

    float **clusters = (float **)malloc(M * sizeof(float *));
    size_t **indices = (size_t **)malloc(M * sizeof(size_t *));

    for (int i = 0; i < M; i++)
    {
        clusters[i] = (float *)malloc(K * TEST_DIM * sizeof(float));
        indices[i] = (size_t *)malloc(TEST_SIZE * sizeof(size_t));
        load(clusters[i], K * TEST_DIM, "./output/cluster" + std::to_string(i));
        load(indices[i], TEST_SIZE, "./output/index" + std::to_string(i));
    }

    productQuantizationQuery(result, input, clusters, indices, input_size, TEST_SIZE, TEST_TOTAL_DIM, M, TOPK);

    for (int i = 0; i < M; i++)
    {
        free(clusters[i]);
        free(indices[i]);
    }
    free(clusters);
    free(indices);
    free(input);

    for (int i = 0; i < input_size * TOPK; i++)
    {
        printf("%ld\t", result[i]);
        if (i != 0 && (i + 1) % TOPK == 0)
        {
            printf("\n");
        }
    }

    free(result);

    return 0;
}