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

    float *input = (float *)malloc(TEST_TOTAL_DIM * sizeof(float));
    size_t *result = (size_t *)malloc(TOPK * sizeof(size_t));
#pragma omp parallel for
    for (int i = 0; i < TEST_TOTAL_DIM; i++)
    {
        input[i] = distrib(gen);
    }
    memset(result, 0, TOPK * sizeof(size_t));

    float **clusters = (float **)malloc(M * sizeof(float *));
    size_t **indices = (size_t **)malloc(M * sizeof(size_t *));
    for (int i = 0; i < M; i++)
    {
        clusters[i] = (float *)malloc(K * TEST_DIM * sizeof(float));
        indices[i] = (size_t *)malloc(TEST_SIZE * sizeof(size_t));
        load(clusters[i], K * TEST_DIM, "cluster" + std::to_string(i));
        load(indices[i], TEST_SIZE, "index" + std::to_string(i));
    }

    query(result, input, clusters, indices, TEST_SIZE, TEST_TOTAL_DIM, M, TOPK);

    for (int i = 0; i < TOPK; i++)
    {
        printf("%ld\t", result[i]);
    }

    for (int i = 0; i < M; i++)
    {
        free(clusters[i]);
        free(indices[i]);
    }
    free(clusters);
    free(indices);
    free(S1);
    free(input);
    free(result);

    return 0;
}