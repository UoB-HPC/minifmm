#pragma once

#include <stdint.h>
#include <stdlib.h>

void seed_rng(uint64_t seed)
{
    srand(seed);
}

double gen_rand()
{
    return (double)rand()/(double)RAND_MAX;
}

double rand_range(double min, double max)
{
    double r = (double)gen_rand();
    return (max-min)*r + min;
}

