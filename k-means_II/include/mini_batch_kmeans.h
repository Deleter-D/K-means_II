#pragma once

#include <stddef.h>
#include "../../utils/include/config.h"
#include "../../utils/include/common.h"
#include "../../utils/include/common.cuh"

void miniBatchInit(float *original_data, size_t original_size, size_t original_dim, float *cluster_set);      // 初始化过程
void miniBatchIteration(float *original_data, size_t original_size, size_t original_dim, float *cluster_set); // 迭代过程
void miniBatchKmeansII(float *original_data, size_t original_size, size_t original_dim, float *cluster_set);