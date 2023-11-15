#pragma once

#define DEBUG_HEAD "\033[34m[Debug]: \033[0m"
#define ERROR_HEAD "\033[31m[Error]: \033[0m"
#define DEBUG

// k-means超参数
#define INIT_ITERATION_TIMES 5
#define K 60000
#define OVER_SAMPLING 2 * K
#define THRESHOLD 0.0001f
#define MAX_KMEANS_ITERATION_TIMES 2000

// 仅用于测试
#define TEST_SIZE 10000
#define TEST_TOTAL_DIM 96
#define TEST_DIM 32
#define TOPK 10
#define M 3