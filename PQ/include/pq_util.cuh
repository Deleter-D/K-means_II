#pragma once

#include <cuda_runtime.h>
#include "../../utils/include/config.h"

__global__ void getAsymmetricDistanceKernel(float *distance, float *distance_tab, unsigned int *index, unsigned int size);
void cudaGetAsymmetricDistance(float *distance, float *distance_tab, unsigned int *index, const unsigned int size);