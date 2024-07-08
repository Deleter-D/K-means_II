#pragma once

#include <cuda_runtime.h>

__global__ void euclideanDistanceKernel(float *distance, float *vec, float *set, const int dim, const int size);
void cudaEuclideanDistance(float *distance, float *vec, float *set, const int dim, const int size);

float cudaCostFromV2S(float *vec, float *cluster_set, const int dim, const unsigned int size);
__global__ void costFromS2SKernel(float *distances, float *original_set, float *cluster_set, int dim, unsigned int original_size, unsigned int cluster_size);
float cudaCostFromS2S(float *original_set, float *cluster_set, const int dim, const unsigned int original_size, const unsigned int cluster_size);

unsigned int cudaBelongV2S(float *x, float *cluster_set, const int dim, const unsigned int size);
__global__ void belongS2SKernel(unsigned int *indices, float *original_set, float *cluster_set, int dim, unsigned int original_size, unsigned int cluster_size);
void cudaBelongS2S(unsigned int *index, float *original_set, float *cluster_set, const int dim, const unsigned int original_size, const unsigned int cluster_size);

__global__ void kmeansppKernel(unsigned int *indices, float *probability, float *cluster_set, float *cluster_final, unsigned int *omega, int dim, int current_k, int cluster_size);
void cudaKmeanspp(float *cluster_final, float *cluster_set, unsigned int *omega, unsigned int k, const int dim, const unsigned int cluster_size);

__global__ void getNewClusterKernel(float *cluster_new, float *original_set, unsigned int *belong, const int dim, const unsigned int original_size, unsigned int *count, unsigned int cluster_size);
void cudaGetNewCluster(float *cluster_new, float *original_set, unsigned int *belong, const int dim, const unsigned int original_size, unsigned int cluster_size);

__global__ void isCloseKernel(float *distance, float *cluster_new, float *cluster_old, float *temp, const int dim, const unsigned int cluster_size);
bool cudaIsClose(float *cluster_new, float *cluster_old, const int dim, const unsigned int cluster_size, float epsilon);