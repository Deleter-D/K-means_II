#pragma once

#include <stddef.h>
#include "../../utils/include/config.h"

void init(float *original_data, unsigned int original_size, unsigned int original_dim, float *cluster_set);      // 初始化过程
void iteration(float *original_data, unsigned int original_size, unsigned int original_dim, float *cluster_set); // 迭代过程
void kmeansII(float *original_data, unsigned int original_size, unsigned int original_dim, float *cluster_set);