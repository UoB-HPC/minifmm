#pragma once

#if defined(__unix__) || defined(__linux__) || defined(__APPLE__)
    #define __FMM_POSIX
    #include <unistd.h>
    #include <stdint.h>
    #include <stdlib.h>
#else
    #include <stdlib.h>
    #include <time.h>
#endif

void seed_rng(uint64_t seed)
{
#ifdef __FMM_POSIX
    srand48(seed);
#else
    srand(seed);
#endif
}

double gen_rand()
{
#ifdef __FMM_POSIX
    return drand48();
#else
    return (double)rand()/(double)RAND_MAX;
#endif
}

double rand_range(double min, double max)
{
    double r = (double)gen_rand();
    return (max-min)*r + min;
}
