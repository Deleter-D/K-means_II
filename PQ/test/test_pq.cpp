#include <random>
#include <cstring>
#include "../include/pq.h"

int main(int argc, char const *argv[])
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> distrib(0, 1);

    float *S1, *input;
    size_t *res;
    S1 = (float *)malloc(TEST_SIZE * TEST_TOTAL_DIM * sizeof(float));
    input = (float *)malloc(TEST_TOTAL_DIM * sizeof(float));
    res = (size_t *)malloc(TOPK * sizeof(size_t));

    for (int i = 0; i < TEST_SIZE * TEST_TOTAL_DIM; i++)
    {
        S1[i] = distrib(gen);
    }
    for (int i = 0; i < TEST_TOTAL_DIM; i++)
    {
        input[i] = distrib(gen);
    }
    memset(res, 0, TOPK * sizeof(size_t));

    build(S1, TEST_SIZE, TEST_TOTAL_DIM, 3);
    query(res, input, TEST_SIZE, TEST_TOTAL_DIM, 3, TOPK);

    free(S1);
    free(input);

    for (int i = 0; i < TOPK; i++)
    {
        printf("%ld\n", res[i]);
    }

    free(res);
    return 0;
}
