#pragma once

#include <cuda_runtime.h>

#include "../../utils/include/config.h"
#include "../../utils/include/common.h"
#include "../../utils/include/common.cuh"
#include "pq_util.cuh"

void build(float *original_data, unsigned int original_size, int dim, unsigned int m, std::string prefix);

void subQuery(float *distance, float *input, float *cluster, unsigned int *index, unsigned int original_size, int dim);
void query(unsigned int *result, float *input, float **clusters, unsigned int **indices, unsigned int original_size, int dim, unsigned int m, unsigned int topk);

extern "C"
{
    void productQuantizationBuild(float *original_data, unsigned int original_size, int original_dim, unsigned int m, char *prefix);
    void productQuantizationQuery(unsigned int *result, float *input, float **clusters, unsigned int **indices, unsigned int input_size, unsigned int original_size, int orininal_dim, unsigned int m, unsigned int topk);
}
