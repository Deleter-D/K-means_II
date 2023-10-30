#include <random>

#include "../include/kmeans.h"

void kmeans::init()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, original_size - 1);

    size_t index = distrib(gen);
    
}

void kmeans::iteration() {}