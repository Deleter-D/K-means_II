#pragma once

#include <cuda_runtime.h>
#include "../../utils/include/config.h"

__global__ void getAsymmetricDistanceKernel(float *distance, float *distance_tab, size_t *index, size_t size);
void getAsymmetricDistance(float *distance, float *distance_tab, size_t *index, const size_t size);