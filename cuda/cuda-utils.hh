#pragma once

#define KOKKOS

template <class T>
KOKKOS_INLINE_FUNCTION void lock(T* val)
{
  while (0 != Kokkos::atomic_compare_exchange(val, 0, 1))
    ;
#ifdef __CUDA_ARCH__
  __threadfence();
#endif
}

template <class T>
KOKKOS_INLINE_FUNCTION void unlock(T* val)
{
#ifdef __CUDA_ARCH__
  __threadfence();
#endif
  while (1 != Kokkos::atomic_compare_exchange(val, 1, 0))
    ;
}

#ifdef __CUDACC__
#define INLINE __device__ __inline__

template <int NTHREADS, int NWORKERS>
class gpu_utils {
public:
  static const int warp_size = 32;
  static const int num_threads = NTHREADS;

  INLINE static int thread_id() { return threadIdx.y; }
  INLINE static int worker_id() { return threadIdx.z; }
  INLINE static int global_worker_id()
  {
    return blockIdx.x * blockDim.z + worker_id();
  }

  template <int N = NTHREADS,
            typename std::enable_if<(N <= 32)>::type* = nullptr>
  INLINE static void sync_worker()
  {
    __syncwarp();
  }

  template <int N = NTHREADS,
            typename std::enable_if<(N > 32)>::type* = nullptr>
  INLINE static void sync_worker()
  {
    __syncthreads();
  }
};
#else
#define INLINE inline
#endif
