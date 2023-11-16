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

float costFromV2S(float *x, float *cluster_set, const int dim, const unsigned int size)
{
    float min = 3.40282347e+38F;
    float temp = 3.40282347e+38F;
    for (unsigned int i = 0; i < size; i++)
    {
        temp = euclideanDistance(x, &cluster_set[i * dim], dim);
        if (temp < min && temp != 0)
            min = temp;
    }
    return min;
}

float costFromS2S(float *original_set, float *cluster_set, const int dim, const unsigned int original_size, const unsigned int cluster_size)
{
    float sum = 0.0f;
    for (unsigned int i = 0; i < original_size; i++)
    {
        sum += costFromV2S(&original_set[i * dim], cluster_set, dim, cluster_size);
    }
    return sum;
}

unsigned int belongV2S(float *x, float *cluster_set, const int dim, const unsigned int size)
{
    float min = 3.40282347e+38F;
    float temp;
    unsigned int index;
    for (unsigned int i = 0; i < size; i++)
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

void belongS2S(unsigned int *index, float *original_set, float *cluster_set, const int dim, const unsigned int original_size, const unsigned int cluster_size)
{
    for (unsigned int i = 0; i < original_size; i++)
    {
        index[i] = belongV2S(&original_set[i * dim], cluster_set, dim, cluster_size);
    }
}

void kmeanspp(float *cluster_final, float *cluster_set, unsigned int *omega, unsigned int k, const int dim, const unsigned int cluster_size)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, cluster_size - 1);

    unsigned int index = distrib(gen);

    // 均匀分布中随机采样一个原聚类中心集的向量放入最终聚类中心集中
    memcpy(&cluster_final[0], &cluster_set[index * dim], dim * sizeof(float));
    unsigned int current_k = 1;

    float max_p;
    float temp_p;
    unsigned int max_p_index;

    // 迭代k-1次，每次取一个聚类中心进入c_final
    while (current_k < k)
    {
        max_p = -1.0f;
        float cost_set2final = costFromS2S(cluster_set, cluster_final, dim, cluster_size, current_k);
        for (unsigned int i = 0; i < cluster_size; i++)
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
void meanVec(float *res, float *original_set, unsigned int *belong, const int dim, const unsigned int original_size, const unsigned int index)
{
    unsigned int count = 0;
    for (unsigned int i = 0; i < original_size; i++)
    {
        if (belong[i] == index)
        {
            for (unsigned int j = 0; j < dim; j++)
            {
                res[j] += original_set[i * dim + j];
            }
            count++;
        }
    }

    for (unsigned int i = 0; i < dim; i++)
    {
        res[i] /= count;
    }
}

bool isClose(float *cluster_new, float *cluster_old, const int dim, const unsigned int cluster_size, float epsilon)
{
    for (unsigned int i = 0; i < cluster_size; i++)
    {
        if (euclideanDistance(&cluster_new[i * dim], &cluster_old[i * dim], dim) > epsilon)
        {
            return false;
        }
    }
    return true;
}

void save(float *data, unsigned int size, const std::string &filename)
{
    std::ofstream outFile(filename, std::ios::out | std::ios::binary);
    if (!outFile)
    {
        std::cerr << ERROR_HEAD << "Can not open file to save." << std::endl;
        return;
    }
    outFile.write(reinterpret_cast<const char *>(data), size * sizeof(float));
    outFile.close();
}

void load(float *data, unsigned int size, const std::string &filename)
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

void save(unsigned int *data, unsigned int size, const std::string &filename)
{
    std::ofstream outFile(filename, std::ios::out | std::ios::binary);
    if (!outFile)
    {
        std::cerr << ERROR_HEAD << "Can not open file to save." << std::endl;
        return;
    }
    outFile.write(reinterpret_cast<const char *>(data), size * sizeof(unsigned int));
    outFile.close();
}

void load(unsigned int *data, unsigned int size, const std::string &filename)
{
    std::ifstream inFile(filename, std::ios::in | std::ios::binary);
    if (!inFile)
    {
        std::cerr << ERROR_HEAD << "Can not open file to load." << std::endl;
        return;
    }
    inFile.read(reinterpret_cast<char *>(data), size * sizeof(unsigned int));
    inFile.close();
}

void split_file(float *original_data, unsigned int original_size, int original_dim, unsigned int m, std::string prefix)
{

    int subset_dim = original_dim / m;
    long int output_file_length = original_size * original_dim * sizeof(float) / m;
    int *output_files = (int *)malloc(m * sizeof(int));
    float **mapped_output_data = (float **)malloc(m * sizeof(float *));
    bool error_flag = false;

#pragma omp parallel for
    for (int i = 0; i < m; i++)
    {
        error_flag = false;
        std::string output_filename(prefix + "subset" + std::to_string(i));
#ifdef DEBUG
        std::cout << DEBUG_HEAD << "subsets will be saved to " << output_filename << "\n";
#endif
        output_files[i] = open(output_filename.c_str(), O_CREAT | O_RDWR | O_TRUNC, 0664);

        if (output_files[i] == -1)
        {
            std::cerr << ERROR_HEAD << "Can not open output file to split." << std::endl;
            error_flag = true;
        }
        truncate(output_filename.c_str(), output_file_length);
        int temp = write(output_files[i], " ", 1);
        if (error_flag || temp == -1)
        {
            close(output_files[i]);
            std::cerr << ERROR_HEAD << "Can not write output file." << std::endl;
            error_flag = true;
        }
        if (!error_flag)
        {
            mapped_output_data[i] = (float *)mmap(NULL, output_file_length, PROT_READ | PROT_WRITE, MAP_SHARED, output_files[i], 0);
            if (mapped_output_data[i] == MAP_FAILED)
            {
                close(output_files[i]);
                std::cerr << ERROR_HEAD << "Can not map output file to memory." << std::endl;
                error_flag = true;
            }
        }
        close(output_files[i]);
    }
#ifdef DEBUG
    std::cout << DEBUG_HEAD << "file created, doing memcpy.\n";
#endif
    if (!error_flag)
    {
#pragma omp parallel for collapse(2)
        for (unsigned int i = 0; i < m; i++)
        {

            for (unsigned int j = 0; j < original_size; j++)
            {
                memcpy(&mapped_output_data[i][j * subset_dim], &original_data[j * original_dim + i * subset_dim], subset_dim * sizeof(float));
            }
        }
#ifdef DEBUG
        std::cout << DEBUG_HEAD << "memcpy finished.\n";
#endif

#pragma omp parallel for
        for (int i = 0; i < m; i++)
        {
            if (munmap(mapped_output_data[i], output_file_length) == -1)
            {
                std::cerr << ERROR_HEAD << "Can not unmap output file from memory." << std::endl;
            }
        }
    }
    free(output_files);
    free(mapped_output_data);
}