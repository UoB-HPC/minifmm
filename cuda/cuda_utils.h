#pragma once

#ifdef __CUDACC__

#define CUDACHK(ans) { gpu_assert((ans), __FILE__, __LINE__); }

inline void gpu_assert(cudaError_t code, const char* file, int line)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "CUDACHK: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

#endif
