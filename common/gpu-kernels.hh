#pragma once

#include <fmm.hh>
#include <node.hh>

#ifdef __CUDACC__

#include <gpu-spharm.hh>

#define NTERMS 4
#warning manually setting nterms

template <class T>
__device__ inline T pow_one(int m)
{
  return static_cast<T>(1 + ((m & 0x01) * -2));
}

template <int TBS, int STILE, int REG, int TPB, int WPB, class T>
__device__ void p2p_gpu(FMM<T>* fmm, node_t<T>* target)
{
#ifdef KOKKOS
  using gpu_utils = gpu_utils<TPB, WPB>;
#endif

  const int tp = target->point_idx;

  __shared__ float4 base_shmem[STILE * WPB];
  float4* shmem = base_shmem + gpu_utils::worker_id() * STILE;

  T ax[REG] = {0};
  T ay[REG] = {0};
  T az[REG] = {0};
  T aw[REG] = {0};
  T xi[REG] = {0};
  T yi[REG] = {0};
  T zi[REG] = {0};

  const int tid = gpu_utils::thread_id();
  const int ilim = ((target->num_points + TBS - 1) / TBS) * TBS;

  for (int ii = tid; ii < ilim; ii += TBS * REG) {
#pragma unroll
    for (int i = 0; i < REG; ++i) {
      ax[i] = 0.0;
      ay[i] = 0.0;
      az[i] = 0.0;
      aw[i] = 0.0;
      xi[i] = ((i * TBS + ii) < target->num_points) ? fmm->x[tp + i * TBS + ii]
                                                    : 0.0;
      yi[i] = ((i * TBS + ii) < target->num_points) ? fmm->y[tp + i * TBS + ii]
                                                    : 0.0;
      zi[i] = ((i * TBS + ii) < target->num_points) ? fmm->z[tp + i * TBS + ii]
                                                    : 0.0;
    }
    for (int jj = 0; jj < target->num_points; jj += STILE) {
      const int jlim = min(STILE, (int)target->num_points - jj);
      gpu_utils::sync_worker();
#pragma unroll
      for (int j = tid; j < jlim; j += TBS) {
        shmem[j] = make_float4(fmm->x[tp + jj + j], fmm->y[tp + jj + j],
                               fmm->z[tp + jj + j], fmm->w[tp + jj + j]);
      }
      gpu_utils::sync_worker();
#pragma unroll
      for (int j = 0; j < jlim; ++j) {
        const float4 sj = shmem[j];
#pragma unroll
        for (int i = 0; i < REG; ++i) {
          const T dx = sj.x - xi[i];
          const T dy = sj.y - yi[i];
          const T dz = sj.z - zi[i];
          const T sw = sj.w;
          const T r = dx * dx + dy * dy + dz * dz;
          const T inv_r = (r == 0.0) ? 0.0 : rsqrtf(r);
          const T inv_r_3 = sw * inv_r * inv_r * inv_r;
          ax[i] += dx * inv_r_3;
          ay[i] += dy * inv_r_3;
          az[i] += dz * inv_r_3;
          aw[i] += sw * inv_r;
        }
      }
    }
#pragma unroll
    for (int i = 0; i < REG; ++i) {
      if ((i * TBS + ii) < target->num_points) {
        // fmm->ax[tp + i*TBS+ii] += ax[i];
        // fmm->ay[tp + i*TBS+ii] += ay[i];
        // fmm->az[tp + i*TBS+ii] += az[i];
        // fmm->p[tp + i*TBS+ii] += aw[i];
        atomicAdd(fmm->ax + tp + i * TBS + ii, ax[i]);
        atomicAdd(fmm->ay + tp + i * TBS + ii, ay[i]);
        atomicAdd(fmm->az + tp + i * TBS + ii, az[i]);
        atomicAdd(fmm->p + tp + i * TBS + ii, aw[i]);
      }
    }
  }
}

template <int TBS, int STILE, int REG, int TPB, int WPB, class T>
__device__ void p2p_gpu(FMM<T>* fmm, node_t<T>* target, node_t<T>* source)
{
#ifdef KOKKOS
  using gpu_utils = gpu_utils<TPB, WPB>;
#endif

  const int tp = target->point_idx;
  const int sp = source->point_idx;

  __shared__ float4 base_shmem[STILE * WPB];
  float4* shmem = base_shmem + gpu_utils::worker_id() * STILE;
  T ax[REG] = {0};
  T ay[REG] = {0};
  T az[REG] = {0};
  T aw[REG] = {0};
  T xi[REG] = {0};
  T yi[REG] = {0};
  T zi[REG] = {0};

  const int tid = gpu_utils::thread_id();
  const int ilim = ((target->num_points + TBS - 1) / TBS) * TBS;

  for (int ii = tid; ii < ilim; ii += TBS * REG) {
#pragma unroll
    for (int i = 0; i < REG; ++i) {
      ax[i] = 0.0;
      ay[i] = 0.0;
      az[i] = 0.0;
      aw[i] = 0.0;
      xi[i] = ((i * TBS + ii) < target->num_points) ? fmm->x[tp + i * TBS + ii]
                                                    : 0.0;
      yi[i] = ((i * TBS + ii) < target->num_points) ? fmm->y[tp + i * TBS + ii]
                                                    : 0.0;
      zi[i] = ((i * TBS + ii) < target->num_points) ? fmm->z[tp + i * TBS + ii]
                                                    : 0.0;
    }
    for (int jj = 0; jj < source->num_points; jj += STILE) {
      const int jlim = min(STILE, (int)source->num_points - jj);
      gpu_utils::sync_worker();
#pragma unroll
      for (int j = tid; j < jlim; j += TBS) {
        shmem[j] = make_float4(fmm->x[sp + jj + j], fmm->y[sp + jj + j],
                               fmm->z[sp + jj + j], fmm->w[sp + jj + j]);
      }
      gpu_utils::sync_worker();
#pragma unroll
      for (int j = 0; j < jlim; ++j) {
        const float4 sj = shmem[j];
#pragma unroll
        for (int i = 0; i < REG; ++i) {
          const T dx = sj.x - xi[i];
          const T dy = sj.y - yi[i];
          const T dz = sj.z - zi[i];
          const T sw = sj.w;
          const T r = dx * dx + dy * dy + dz * dz;
          const T inv_r = rsqrtf(r);
          const T inv_r_3 = sw * inv_r * inv_r * inv_r;
          ax[i] += dx * inv_r_3;
          ay[i] += dy * inv_r_3;
          az[i] += dz * inv_r_3;
          aw[i] += sw * inv_r;
        }
      }
    }
#pragma unroll
    for (int i = 0; i < REG; ++i) {
      if ((i * TBS + ii) < target->num_points) {
        // fmm->ax[tp + i*TBS+ii] += ax[i];
        // fmm->ay[tp + i*TBS+ii] += ay[i];
        // fmm->az[tp + i*TBS+ii] += az[i];
        // fmm->p[tp + i*TBS+ii] += aw[i];
        atomicAdd(fmm->ax + tp + i * TBS + ii, ax[i]);
        atomicAdd(fmm->ay + tp + i * TBS + ii, ay[i]);
        atomicAdd(fmm->az + tp + i * TBS + ii, az[i]);
        atomicAdd(fmm->p + tp + i * TBS + ii, aw[i]);
      }
    }
  }
}

template <int TPB, int WPB, class T>
__device__ void m2l_gpu(FMM<T>* fmm, node_t<T>* target, node_t<T>* source)
{
#ifdef KOKKOS
  using gpu_utils = gpu_utils<TPB, WPB>;
#endif

  const int size = NTERMS * NTERMS;
  const int shmem_size = size * (sizeof(T) + sizeof(complex_t<T>) * 2);
  __shared__ char shmem[shmem_size * WPB];
  char* warp_shmem = shmem + gpu_utils::worker_id() * shmem_size;

  T* legendre = (T*)warp_shmem;
  complex_t<T>* outer = (complex_t<T>*)(warp_shmem + sizeof(T) * size);
  complex_t<T>* shared_m =
      (complex_t<T>*)(warp_shmem + (sizeof(complex_t<T>) + sizeof(T)) * size);

  T dx = target->cx - source->cx;
  T dy = target->cy - source->cy;
  T dz = target->cz - source->cz;
  T rho, alpha, beta;
  cart_to_sph(dx, dy, dz, rho, alpha, beta);
  compute_outer_gpu<TPB, WPB>(fmm, rho, alpha, beta, outer, legendre);
  complex_t<T>* Msource = &fmm->m[source->mult_idx];
  complex_t<T>* Ltarget = &fmm->l[target->mult_idx];

#pragma unroll
  for (int i = gpu_utils::thread_id(); i < fmm->num_terms * fmm->num_terms;
       i += TPB) {
    shared_m[i] = Msource[i];
  }
  gpu_utils::sync_worker();

#pragma unroll
  for (int i = gpu_utils::thread_id(); i < fmm->num_terms * fmm->num_terms;
       i += TPB) {
    const int j = (int)sqrtf((float)i);
    const int k = i - j * j - j;

    complex_t<T> tmp(0.0, 0.0);
    for (int n = 0; n < fmm->num_terms - j; ++n) {
      for (int m = -n; m <= n; ++m) {
        tmp += shared_m[mult_idx(n, m)] * outer[mult_idx(j + n, -k - m)];
      }
    }
    // Ltarget[i] += tmp;
    atomicAdd(&(Ltarget[i].re), tmp.re);
    atomicAdd(&(Ltarget[i].im), tmp.im);
  }
}

template <int TPB, int WPB, class T>
INLINE void p2m_gpu(FMM<T>* fmm, node_t<T>* node)
{
#ifdef KOKKOS
  using gpu_utils = gpu_utils<TPB, WPB>;
#endif

  size_t pt_offset = node->point_idx;
  size_t mt_offset = node->mult_idx;
 
  __shared__ T shmem_all[WPB * NTERMS * NTERMS * 3];
  T* shmem = shmem_all + gpu_utils::worker_id() * NTERMS * NTERMS * 3;
  T* legendre = shmem;
  complex_t<T>* inner = (complex_t<T>*)(shmem + NTERMS * NTERMS);

  for (size_t i = 0; i < node->num_points; ++i) {
    T dx = fmm->x[i + pt_offset] - node->cx;
    T dy = fmm->y[i + pt_offset] - node->cy;
    T dz = fmm->z[i + pt_offset] - node->cz;
    T r, theta, phi;
    cart_to_sph(dx, dy, dz, r, theta, phi);
    compute_inner_gpu<TPB, WPB>(fmm, r, theta, phi, inner, legendre);
    gpu_utils::sync_worker();
#pragma unroll
    for (int t = gpu_utils::thread_id(); t < fmm->num_terms * fmm->num_terms;
         t += TPB) {
      const int n = (int)sqrtf((float)t);
      fmm->m[mt_offset + t] += fmm->w[i + pt_offset] * pow_one<T>(n) * inner[t];
    }
  }
}

template <int TPB, int WPB, class T>
__device__ void m2m_gpu(FMM<T>* fmm, node_t<T>* node)
{
#ifdef KOKKOS
  using gpu_utils = gpu_utils<TPB, WPB>;
#endif

  __shared__ T shmem_all[WPB * NTERMS * NTERMS * 3];
  T* shmem = shmem_all + gpu_utils::worker_id() * NTERMS * NTERMS * 3;
  T* legendre = shmem;
  complex_t<T>* inner = (complex_t<T>*)(shmem + NTERMS * NTERMS);

  for (size_t i = 0; i < node->num_children; ++i) {
    node_t<T>* child = &fmm->nodes[node->child[i]];
    T dx = node->cx - child->cx;
    T dy = node->cy - child->cy;
    T dz = node->cz - child->cz;
    T r, theta, phi;
    cart_to_sph(dx, dy, dz, r, theta, phi);

    const complex_t<T>* Mchild = &fmm->m[child->mult_idx];
    complex_t<T>* Mnode = &fmm->m[node->mult_idx];

    compute_inner_gpu<TPB, WPB>(fmm, r, theta, phi, inner, legendre);
    gpu_utils::sync_worker();
#pragma unroll
    for (int t = gpu_utils::thread_id(); t < fmm->num_terms * fmm->num_terms;
         t += TPB) {
      const int j = (int)sqrtf((float)t);
      const int k = t - j * j - j;
      complex_t<T> tmp(static_cast<T>(0.0), static_cast<T>(0.0));
      for (int n = 0; n <= j; ++n) {
        for (int m = -n; m <= n; ++m) {
          if (abs(k - m) <= j - n)
            tmp += Mchild[mult_idx(n, m)] * inner[mult_idx(j - n, k - m)];
        }
      }
      Mnode[t] += tmp;
    }
  }
}

#endif
