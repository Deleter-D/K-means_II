#include <random>
#include <cstring>
#include <omp.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include "../include/vq.cuh"

#define INPUT_SIZE 10000000

int main(int argc, char const *argv[])
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> distrib(0, 1);
    std::normal_distribution<float> normal(5, 10);

    int fd = open("/mnt/data/Baidu/test/initdata/mini_original_data", O_RDONLY);
    if (fd == -1)
    {
        std::cout << "open faild.\n";
        return 0;
    }
    struct stat sb;
    if (fstat(fd, &sb) == -1)
    {
        std::cout << "get size faild.\n";
        return 0;
    }
    float *S1 = (float *)mmap(NULL, sb.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (S1 == MAP_FAILED)
    {
        std::cout << "map faild.\n";
        return 0;
    }

    VQBuild(S1, INPUT_SIZE, TEST_TOTAL_DIM, "./output");

    munmap(S1, sb.st_size);

    unsigned int input_size = 10;
    float *input = (float *)malloc(input_size * TEST_TOTAL_DIM * sizeof(float));
    unsigned int *result = (unsigned int *)malloc(input_size * TOPK * sizeof(unsigned int));
#pragma omp parallel for
    for (int i = 0; i < input_size * TEST_TOTAL_DIM; i++)
    {
        input[i] = normal(gen);
    }
    memset(result, 0, input_size * TOPK * sizeof(unsigned int));

    float *cluster_0 = (float *)malloc(K0 * TEST_TOTAL_DIM * sizeof(float));
    float *cluster_1 = (float *)malloc(K1 * TEST_TOTAL_DIM * sizeof(float));
    unsigned int *indices_0 = (unsigned int *)malloc(INPUT_SIZE * sizeof(unsigned int));
    unsigned int *indices_1 = (unsigned int *)malloc(K0 * sizeof(unsigned int));
    unsigned int *catagories_count = (unsigned int *)malloc(K0 * sizeof(unsigned int));
    load(cluster_0, K0 * TEST_TOTAL_DIM, "./output/cluster_0");
    load(cluster_1, K1 * TEST_TOTAL_DIM, "./output/cluster_1");
    load(indices_0, INPUT_SIZE, "./output/indices_0");
    load(indices_1, K0, "./output/indices_1");
    load(catagories_count, K0, "./output/catagories");

    VQQuery(result, input, input_size, S1, TEST_TOTAL_DIM, INPUT_SIZE, TOPK, cluster_0, cluster_1, indices_0, indices_1, catagories_count);

    free(cluster_0);
    free(cluster_1);
    free(indices_0);
    free(indices_1);
    free(catagories_count);
    free(input);

    for (int i = 0; i < input_size * TOPK; i++)
    {
        printf("%d\t", result[i]);
        if (i != 0 && (i + 1) % TOPK == 0)
        {
            printf("\n");
        }
    }

    free(result);

    return 0;
}