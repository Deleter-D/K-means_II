#pragma once

#include <stddef.h>
#include "../../utils/include/config.h"
#include "../../utils/include/common.h"
#include "../../utils/include/common.cuh"

void randomInit(float *original_data, unsigned int original_size, unsigned int original_dim, float *cluster_set, int cluster_size);      // 初始化过程
void randomIteration(float *original_data, unsigned int original_size, unsigned int original_dim, float *cluster_set, int cluster_size); // 迭代过程
void randomKmeans(float *original_data, unsigned int original_size, unsigned int original_dim, float *cluster_set, int cluster_size);