#pragma once

#include "../../utils/include/config.h"
#include "../../utils/include/common.h"
#include "../../utils/include/common.cuh"
#include "../../k-means/include/random_kmeans.h"

void build(float *original_data, unsigned int original_size, int original_dim, char *prefix);