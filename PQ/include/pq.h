#pragma once

#include <cuda_runtime.h>

#include "../../utils/include/config.h"
#include "../../utils/include/common.h"
#include "../../utils/include/common.cuh"
#include "pq_util.cuh"

void build(float *original_data, size_t original_size, int original_dim, unsigned int m);

void query(size_t *result, float *input, size_t original_size, int original_dim, unsigned int m, unsigned int topk);
