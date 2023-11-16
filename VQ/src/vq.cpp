#include "../include/vq.h"

void build(float *original_data, unsigned int original_size, int original_dim, char *prefix)
{
    float *cluster_0 = (float *)malloc(K0 * original_dim * sizeof(float));
    randomKmeans(original_data, original_size, original_dim, cluster_0, K0);
}