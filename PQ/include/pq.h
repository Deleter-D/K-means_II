#pragma once

#include <cuda_runtime.h>

#include "../../utils/include/config.h"
#include "../../utils/include/common.h"
#include "../../utils/include/common.cuh"
#include "pq_util.cuh"

void build(float *original_data, size_t original_size, int original_dim, unsigned int m);

void subQuery(float *distance, float *input, float *cluster, size_t *index, size_t original_size, int dim);
void query(size_t *result, float *input, float **clusters, size_t **indices, size_t original_size, int dim, unsigned int m, unsigned int topk);

void productQuantizationBuild(std::string &filename, size_t original_size, int original_dim, unsigned int m);
void productQuantizationQuery(size_t *result, float *input, size_t input_size, size_t original_size, int dim, unsigned int m, unsigned int topk);
