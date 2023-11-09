#include <math.h>
#include <random>
#include <cstring>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include "../include/common.h"

float euclideanDistance(float *x, float *y, const int dim)
{
    float sum = 0.0f;
    float difference = 0.0f;
    for (int i = 0; i < dim; i++)
    {
        difference = x[i] - y[i];
        sum += difference * difference;
    }
    return sum;
}

float costFromV2S(float *x, float *cluster_set, const int dim, const size_t size)
{
    float min = MAXFLOAT;
    float temp = MAXFLOAT;
    for (size_t i = 0; i < size; i++)
    {
        temp = euclideanDistance(x, &cluster_set[i * dim], dim);
        if (temp < min)
            min = temp;
    }
    return min;
}

float costFromS2S(float *original_set, float *cluster_set, const int dim, const size_t original_size, const size_t cluster_size)
{
    float sum = 0.0f;
    for (size_t i = 0; i < original_size; i++)
    {
        sum += costFromV2S(&original_set[i * dim], cluster_set, dim, cluster_size);
    }
    return sum;
}

size_t belongV2S(float *x, float *cluster_set, const int dim, const size_t size)
{
    float min = MAXFLOAT;
    float temp;
    size_t index;
    for (size_t i = 0; i < size; i++)
    {
        temp = euclideanDistance(x, &cluster_set[i * dim], dim);
        if (temp < min)
        {
            min = temp;
            index = i;
        }
    }
    return index;
}

void belongS2S(size_t *index, float *original_set, float *cluster_set, const int dim, const size_t original_size, const size_t cluster_size)
{
    for (size_t i = 0; i < original_size; i++)
    {
        index[i] = belongV2S(&original_set[i * dim], cluster_set, dim, cluster_size);
    }
}

void kmeanspp(float *cluster_final, float *cluster_set, size_t *omega, size_t k, const int dim, const size_t cluster_size)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, cluster_size - 1);

    size_t index = distrib(gen);

    // 均匀分布中随机采样一个原聚类中心集的向量放入最终聚类中心集中
    memcpy(&cluster_final[0], &cluster_set[index * dim], dim * sizeof(float));
    size_t current_k = 1;

    float max_p;
    float temp_p;
    size_t max_p_index;

    // 迭代k-1次，每次取一个聚类中心进入c_final
    while (current_k < k)
    {
        max_p = -1.0f;
        float cost_set2final = costFromS2S(cluster_set, cluster_final, dim, cluster_size, current_k);
        for (size_t i = 0; i < cluster_size; i++)
        {
            // 计算当前向量的概率
            temp_p = omega[i] * costFromV2S(&cluster_set[i * dim], cluster_final, dim, current_k) / cost_set2final;
            // 记录概率最大的向量信息
            if (temp_p > max_p)
            {
                max_p = temp_p;
                max_p_index = i;
            }
        }
        // 将概率最大的向量并入最终聚类中心集
        memcpy(&cluster_final[current_k * dim], &cluster_set[max_p_index * dim], dim * sizeof(float));
        current_k++;
    }
}
void meanVec(float *res, float *original_set, size_t *belong, const int dim, const size_t original_size, const size_t index)
{
    size_t count = 0;
    for (size_t i = 0; i < original_size; i++)
    {
        if (belong[i] == index)
        {
            for (size_t j = 0; j < dim; j++)
            {
                res[j] += original_set[i * dim + j];
            }
            count++;
        }
    }

    for (size_t i = 0; i < dim; i++)
    {
        res[i] /= count;
    }
}

bool isClose(float *cluster_new, float *cluster_old, const int dim, const size_t cluster_size, float epsilon)
{
    for (size_t i = 0; i < cluster_size; i++)
    {
        if (euclideanDistance(&cluster_new[i * dim], &cluster_old[i * dim], dim) > epsilon)
        {
            return false;
        }
    }
    return true;
}

void save(float *data, size_t size, const std::string &filename)
{
    std::ofstream outFile(filename, std::ios::out | std::ios::binary | std::ios::app);
    if (!outFile)
    {
        std::cerr << ERROR_HEAD << "Can not open file to save." << std::endl;
        return;
    }
    outFile.write(reinterpret_cast<const char *>(data), size * sizeof(float));
    outFile.close();
}

void load(float *data, size_t size, const std::string &filename)
{
    std::ifstream inFile(filename, std::ios::in | std::ios::binary);
    if (!inFile)
    {
        std::cerr << ERROR_HEAD << "Can not open file to load." << std::endl;
        return;
    }
    inFile.read(reinterpret_cast<char *>(data), size * sizeof(float));
    inFile.close();
}

void save(size_t *data, size_t size, const std::string &filename)
{
    std::ofstream outFile(filename, std::ios::out | std::ios::binary | std::ios::app);
    if (!outFile)
    {
        std::cerr << ERROR_HEAD << "Can not open file to save." << std::endl;
        return;
    }
    outFile.write(reinterpret_cast<const char *>(data), size * sizeof(size_t));
    outFile.close();
}

void load(size_t *data, size_t size, const std::string &filename)
{
    std::ifstream inFile(filename, std::ios::in | std::ios::binary);
    if (!inFile)
    {
        std::cerr << ERROR_HEAD << "Can not open file to load." << std::endl;
        return;
    }
    inFile.read(reinterpret_cast<char *>(data), size * sizeof(size_t));
    inFile.close();
}

void split_file(const std::string &filename, size_t size, int dim, unsigned int m)
{
    int input_file = open(filename.c_str(), O_RDWR);
    struct stat input_file_sb;

    if (input_file == -1)
    {
        std::cerr << ERROR_HEAD << "Can not open file to split." << std::endl;
        return;
    }

    if (fstat(input_file, &input_file_sb) == -1)
    {
        std::cerr << ERROR_HEAD << "Can not get file size." << std::endl;
        return;
    }
    float *mapped_data = (float *)mmap(NULL, input_file_sb.st_size, PROT_READ | PROT_WRITE, MAP_SHARED, input_file, 0);
    if (mapped_data == MAP_FAILED)
    {
        close(input_file);
        std::cerr << ERROR_HEAD << "Can not map file to memory." << std::endl;
        return;
    }
    close(input_file);

    int subset_dim = dim / m;
    long int output_file_length = input_file_sb.st_size / m;
    int *output_files = (int *)malloc(m * sizeof(int));
    float **mapped_output_data = (float **)malloc(m * sizeof(float *));

#pragma omp parallel for
    for (int i = 0; i < m; i++)
    {
        std::string output_filename("subset" + std::to_string(i));
        output_files[i] = open(output_filename.c_str(), O_CREAT | O_RDWR | O_TRUNC, 0664);
        if (output_files[i] == -1)
        {
            std::cerr << ERROR_HEAD << "Can not open output file to split." << std::endl;
            return;
        }
        truncate(output_filename.c_str(), output_file_length);
        int temp = write(output_files[i], " ", 1);
        if (temp == -1)
        {
            close(output_files[i]);
            std::cerr << ERROR_HEAD << "Can not write output file." << std::endl;
            return;
        }
        mapped_output_data[i] = (float *)mmap(NULL, output_file_length, PROT_READ | PROT_WRITE, MAP_SHARED, output_files[i], 0);
        if (mapped_output_data[i] == MAP_FAILED)
        {
            close(output_files[i]);
            std::cerr << ERROR_HEAD << "Can not map output file to memory." << std::endl;
            return;
        }
        close(output_files[i]);
    }
#pragma omp parallel for collapse(2)
    for (unsigned int i = 0; i < m; i++)
    {
        for (size_t j = 0; j < size; j++)
        {
            memcpy(&mapped_output_data[i][j * subset_dim], &mapped_data[j * dim + i * subset_dim], subset_dim * sizeof(float));
        }
    }

#pragma omp parallel for
    for (int i = 0; i < m; i++)
    {
        if (munmap(mapped_output_data[i], output_file_length) == -1)
        {
            std::cerr << ERROR_HEAD << "Can not unmap output file from memory." << std::endl;
        }
    }

    if (munmap(mapped_data, input_file_sb.st_size) == -1)
    {
        std::cerr << ERROR_HEAD << "Can not unmap file from memory." << std::endl;
    }

    free(output_files);
    free(mapped_output_data);
}