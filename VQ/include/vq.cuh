#pragma once

#include "../../utils/include/config.h"
#include "../../utils/include/common.h"
#include "../../utils/include/common.cuh"
#include "../../k-means/include/random_kmeans.h"

void VQBuild(float *original_data, unsigned int original_size, int original_dim, char *prefix);

__global__ void levelQueryKernel(unsigned int *result, float *query_set, float *cluster, unsigned int *indices, unsigned int *belong,
                                 unsigned int dim, unsigned int query_size, unsigned int cluster_size, unsigned int indices_size);
void VQQuery(unsigned int *result, float *query_set, unsigned int query_size,
             float *original_data, unsigned int original_dim, unsigned int original_size, unsigned int topk,
             float *cluster_0, float *cluster_1, unsigned int *indices_0, unsigned int *indices_1, unsigned int *catagories_count);