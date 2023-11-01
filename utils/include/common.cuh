#pragma once

#include <cuda_runtime.h>

__global__ void euclideanDistanceKernel(float *distance, float *vec, float *set, float *temp, const int dim, const int size);
float *cudaEuclideanDistance(float *vec, float *set, const int dim, const int size);

float cudaCostFromV2S(float *vec, float *cluster_set, const int dim, const size_t size);
float cudaCostFromS2S(float *original_set, float *cluster_set, const int dim, const size_t original_size, const size_t cluster_size);

size_t cudaBelongV2S(float *x, float *cluster_set, const int dim, const size_t size);
size_t *cudaBelongS2S(float *original_set, float *cluster_set, const int dim, const size_t original_size, const size_t cluster_size);

float *cudaKmeanspp(float *cluster_set, size_t *omega, size_t k, const int dim, const size_t cluster_size);