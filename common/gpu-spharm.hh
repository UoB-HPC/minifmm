#pragma once

#include <spharm.hh>

template <int TPB, int WPB, class T>
INLINE void compute_outer_gpu(FMM<T>* fmm, T r, T theta, T phi,
                                  complex_t<T>* outer, T* legendre,
                                  complex_t<T>* outer_deriv = nullptr,
                                  T* legendre_deriv = nullptr)
{
#ifdef KOKKOS
  using gpu_utils = gpu_utils<TPB, WPB>;
#endif
  if (fmm->num_terms == 0) return;
  const int lid = gpu_utils::thread_id();
  const int num_lanes = TPB;
  if (lid == 0) {
    compute_legendre<1>(fmm->num_terms - 1, std::cos(theta), legendre,
                            legendre_deriv);
  }
  gpu_utils::sync_worker();
#pragma unroll
  for (int i = lid; i < fmm->num_terms * fmm->num_terms; i += num_lanes) {
    const int n = (int)sqrtf((float)i);
    const int m = i - n * n - n;
    outer[i] = fmm->outer_factors[i] * legendre[leg_idx(n, m)] *
               complex_exp(complex_t<T>(0.0, 1.0) * static_cast<T>(m) * phi) *
               (static_cast<T>(1.0) / std::pow(r, static_cast<T>(n + 1)));
  }
}

template <int TPB, int WPB, class T>
INLINE void compute_inner_gpu(FMM<T>* fmm, T r, T theta, T phi,
                                  complex_t<T>* inner, T* legendre,
                                  complex_t<T>* inner_deriv = nullptr,
                                  T* legendre_deriv = nullptr)
{
#ifdef KOKKOS
  using gpu_utils = gpu_utils<TPB, WPB>;
#endif
  if (fmm->num_terms == 0) return;
  const int lid = gpu_utils::thread_id();
  const int num_lanes = TPB;
  if (lid == 0) {
    compute_legendre<1>(fmm->num_terms - 1, std::cos(theta), legendre,
                            legendre_deriv);
  }
  gpu_utils::sync_worker();
#pragma unroll
  for (int i = lid; i < fmm->num_terms * fmm->num_terms; i += num_lanes) {
    const int n = (int)sqrtf((float)i);
    const int m = i - n * n - n;
    inner[i] = fmm->inner_factors[i] * legendre[leg_idx(n, m)] *
               complex_exp(complex_t<T>(0.0, 1.0) * static_cast<T>(m) * phi) *
               std::pow(r, static_cast<T>(n));
  }
}

