#pragma once

#ifdef __CUDACC__

#define HOSTDEVICE __host__ __device__
#define CUDACHK(ans)                                                           \
  {                                                                            \
    gpu_assert((ans), __FILE__, __LINE__);                                     \
  }
inline void gpu_assert(cudaError_t code, const char* file, int line)
{
  if (code != cudaSuccess) {
    fprintf(stderr, "CUDACHK: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    exit(code);
  }
}
#else
#define HOSTDEVICE
#endif

template <class T>
struct FMM;

template <class T>
void alloc_and_copy(T** dst, const T* src, size_t nelms)
{
#ifdef __CUDACC__
  CUDACHK(cudaMalloc((void**)dst, sizeof(T) * nelms));
  CUDACHK(cudaMemcpy(*dst, src, sizeof(T) * nelms, cudaMemcpyHostToDevice));
#else
  *dst = (T*)malloc(sizeof(T) * nelms);
  memcpy(*dst, src, sizeof(T) * nelms);
#endif
}

template <class T>
void copy_back(T* dst, T* src, size_t nelms)
{
#ifdef __CUDACC__
  CUDACHK(cudaMemcpy(dst, src, sizeof(T) * nelms, cudaMemcpyDeviceToHost));
#else
  memcpy(dst, src, sizeof(T) * nelms);
#endif
}

template <class T>
void init_device_fmm(FMM<T>* fmm, FMM<T>** h_fmm_ret, FMM<T>** d_fmm_ret)
{
  FMM<T>* h_fmm = (FMM<T>*)malloc(sizeof(FMM<T>));
  *h_fmm = *fmm;

  alloc_and_copy(&h_fmm->nodes, fmm->nodes, fmm->num_nodes);
  alloc_and_copy(&h_fmm->x, fmm->x, fmm->num_points);
  alloc_and_copy(&h_fmm->y, fmm->y, fmm->num_points);
  alloc_and_copy(&h_fmm->z, fmm->z, fmm->num_points);
  alloc_and_copy(&h_fmm->w, fmm->w, fmm->num_points);
  alloc_and_copy(&h_fmm->ax, fmm->ax, fmm->num_points);
  alloc_and_copy(&h_fmm->ay, fmm->ay, fmm->num_points);
  alloc_and_copy(&h_fmm->az, fmm->az, fmm->num_points);
  alloc_and_copy(&h_fmm->p, fmm->p, fmm->num_points);
  alloc_and_copy(&h_fmm->inner_factors, fmm->inner_factors,
                 fmm->num_multipoles);
  alloc_and_copy(&h_fmm->outer_factors, fmm->outer_factors,
                 fmm->num_multipoles);
  alloc_and_copy(&h_fmm->m, fmm->m, fmm->num_multipoles * fmm->num_nodes);
  alloc_and_copy(&h_fmm->l, fmm->l, fmm->num_multipoles * fmm->num_nodes);

  FMM<T>* d_fmm;
  alloc_and_copy(&d_fmm, h_fmm, 1);

  *h_fmm_ret = h_fmm;
  *d_fmm_ret = d_fmm;
}

template <class T>
void device_free(T* p)
{
#ifdef __CUDACC__
  CUDACHK(cudaFree(p));
#else
  free(p);
#endif
}

template <class T>
void fini_device_fmm(FMM<T>* fmm, FMM<T>* h_fmm, FMM<T>* d_fmm)
{
  copy_back(fmm->ax, h_fmm->ax, fmm->num_points);
  copy_back(fmm->ay, h_fmm->ay, fmm->num_points);
  copy_back(fmm->az, h_fmm->az, fmm->num_points);
  copy_back(fmm->p, h_fmm->p, fmm->num_points);

  copy_back(fmm->m, h_fmm->m, fmm->num_nodes * fmm->num_multipoles);
  copy_back(fmm->l, h_fmm->l, fmm->num_nodes * fmm->num_multipoles);

  device_free(h_fmm->nodes);
  device_free(h_fmm->x);
  device_free(h_fmm->y);
  device_free(h_fmm->z);
  device_free(h_fmm->w);
  device_free(h_fmm->ax);
  device_free(h_fmm->ay);
  device_free(h_fmm->az);
  device_free(h_fmm->p);
  device_free(h_fmm->inner_factors);
  device_free(h_fmm->outer_factors);
  device_free(h_fmm->m);
  device_free(h_fmm->l);

  device_free(d_fmm);

  free(h_fmm);
}

template <class T>
void update_device_array(T* d_array, T* h_array, size_t n)
{
#ifdef __CUDACC__
  CUDACHK(cudaMemcpy(d_array, h_array, sizeof(T) * n, cudaMemcpyHostToDevice));
#else
  memcpy(d_array, h_array, sizeof(T) * n);
#endif
}

template <class T>
void update_host_array(T* h_array, T* d_array, size_t n)
{
#ifdef __CUDACC__
  CUDACHK(cudaMemcpy(h_array, d_array, sizeof(T) * n, cudaMemcpyDeviceToHost));
#else
  memcpy(h_array, d_array, sizeof(T) * n);
#endif
}
