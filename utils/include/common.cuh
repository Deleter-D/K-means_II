#pragma once

#include <cuda_runtime.h>

__global__ void euclideanDistanceKernel(float *distance, float *vec, float *set, float *temp, const int dim, const int size);
void cudaEuclideanDistance(float *distance, float *vec, float *set, const int dim, const int size);

float cudaCostFromV2S(float *vec, float *cluster_set, const int dim, const size_t size);
__global__ void costFromS2SKernel(float *distances, float *original_set, float *cluster_set, int dim, size_t original_size, size_t cluster_size);
float cudaCostFromS2S(float *original_set, float *cluster_set, const int dim, const size_t original_size, const size_t cluster_size);

size_t cudaBelongV2S(float *x, float *cluster_set, const int dim, const size_t size);
__global__ void belongS2SKernel(size_t *indices, float *distances, float *original_set, float *cluster_set, int dim, size_t original_size, size_t cluster_size);
void cudaBelongS2S(size_t *index, float *original_set, float *cluster_set, const int dim, const size_t original_size, const size_t cluster_size);

__global__ void kmeansppKernel(size_t *indices, float *probability, float *cluster_set, float *cluster_final, size_t *omega, int dim, int current_k, int cluster_size);
void cudaKmeanspp(float *cluster_final, float *cluster_set, size_t *omega, size_t k, const int dim, const size_t cluster_size);

__global__ void getNewClusterKernel(float *cluster_new, float *original_set, size_t *belong, const int dim, const size_t original_size, unsigned int *count);
void cudaGetNewCluster(float *cluster_new, float *original_set, size_t *belong, const int dim, const size_t original_size);

__global__ void isCloseKernel(float *distance, float *cluster_new, float *cluster_old, float *temp, const int dim, const size_t cluster_size);
bool cudaIsClose(float *cluster_new, float *cluster_old, const int dim, const size_t cluster_size, float epsilon);