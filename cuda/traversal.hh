#pragma once

#include <omp.h>

#include <get-deps.hh>
#include <kernels.hh>
#include <node.hh>

#define INLINE __device__ __inline__

// for compatability with generic gpu kernels
namespace gpu_utils {
INLINE static int thread_id() { return threadIdx.x; }
INLINE static int worker_id() { return 0; }
INLINE static void sync_worker() { __syncthreads(); }
}  // namespace gpu_utils

#include <gpu-kernels.hh>

template <class T>
void upwards_pass(FMM<T>* fmm, node_t<T>* node)
{
  for (size_t i = 0; i < node->num_children; ++i) {
#pragma omp task
    upwards_pass(fmm, &fmm->nodes[node->child[i]]);
  }
#pragma omp taskwait

  if (node->is_leaf())
    p2m(fmm, node);
  else
    m2m(fmm, node);
}

template <class T>
void downwards_pass(FMM<T>* fmm, node_t<T>* node)
{
  if (node->is_leaf())
    l2p(fmm, node);
  else {
    l2l(fmm, node);
    for (size_t i = 0; i < node->num_children; ++i) {
#pragma omp task
      downwards_pass(fmm, &fmm->nodes[node->child[i]]);
    }
  }
#pragma omp taskwait
}

template <class T>
__global__ void p2p_kernel(FMM<T>* d_fmm, size_t* d_p2p_nodes,
                           size_t* d_p2p_deps_array, size_t* d_p2p_deps_offsets,
                           size_t* d_p2p_deps_sizes)
{
  const int i = blockIdx.x;
  node_t<T>* target = d_fmm->nodes + d_p2p_nodes[i];
  size_t num_deps = d_p2p_deps_sizes[i];
  size_t offset = d_p2p_deps_offsets[i];
  for (size_t j = 0; j < num_deps; ++j) {
    size_t source_idx = d_p2p_deps_array[offset + j];
    node_t<T>* source = d_fmm->nodes + source_idx;
    if (target == source)
      p2p_gpu<128, 128, 1, 128, 1>(d_fmm, target);
    else
      p2p_gpu<128, 128, 1, 128, 1>(d_fmm, target, source);
  }
}

template <class T>
__global__ void m2l_kernel(FMM<T>* d_fmm, size_t* d_m2l_nodes,
                           size_t* d_m2l_deps_array, size_t* d_m2l_deps_offsets,
                           size_t* d_m2l_deps_sizes)
{
  const int i = blockIdx.x;
  node_t<T>* target = d_fmm->nodes + d_m2l_nodes[i];
  size_t num_deps = d_m2l_deps_sizes[i];
  size_t offset = d_m2l_deps_offsets[i];
  for (size_t j = 0; j < num_deps; ++j) {
    size_t source_idx = d_m2l_deps_array[offset + j];
    node_t<T>* source = d_fmm->nodes + source_idx;
    m2l_gpu<32, 1>(d_fmm, target, source);
  }
}

template <class T>
void cuda_dtt(FMM<T>* fmm, FMM<T>* h_fmm, FMM<T>* d_fmm)
{
  Timer timer;

  timer.start();

  std::vector<std::vector<size_t>> p2p_deps(fmm->num_nodes);
  std::vector<std::vector<size_t>> m2l_deps(fmm->num_nodes);

  get_deps_omp(fmm, &p2p_deps, &m2l_deps);
  timer.stop();
  printf("    %-16s %12.8f\n", "Deps. Time (s) ", timer.elapsed());

  timer.start();

  size_t* h_p2p_nodes;
  size_t* h_p2p_deps_array;
  size_t* h_p2p_deps_offsets;
  size_t* h_p2p_deps_sizes;
  size_t* d_p2p_nodes;
  size_t* d_p2p_deps_array;
  size_t* d_p2p_deps_offsets;
  size_t* d_p2p_deps_sizes;
  size_t p2p_deps_tot;
  size_t p2p_num_nodes;

  size_t* h_m2l_nodes;
  size_t* h_m2l_deps_array;
  size_t* h_m2l_deps_offsets;
  size_t* h_m2l_deps_sizes;
  size_t* d_m2l_nodes;
  size_t* d_m2l_deps_array;
  size_t* d_m2l_deps_offsets;
  size_t* d_m2l_deps_sizes;
  size_t m2l_deps_tot;
  size_t m2l_num_nodes;

  pack_deps(p2p_deps, &h_p2p_nodes, &h_p2p_deps_array, &h_p2p_deps_offsets,
            &h_p2p_deps_sizes, &p2p_deps_tot, &p2p_num_nodes);
  pack_deps(m2l_deps, &h_m2l_nodes, &h_m2l_deps_array, &h_m2l_deps_offsets,
            &h_m2l_deps_sizes, &m2l_deps_tot, &m2l_num_nodes);
  timer.stop();
  printf("%-20s %12.8f\n", "  Pack Time (s) ", timer.elapsed());

  timer.start();
  alloc_and_copy(&d_p2p_nodes, h_p2p_nodes, p2p_num_nodes);
  alloc_and_copy(&d_p2p_deps_array, h_p2p_deps_array, p2p_deps_tot);
  alloc_and_copy(&d_p2p_deps_offsets, h_p2p_deps_offsets, p2p_num_nodes);
  alloc_and_copy(&d_p2p_deps_sizes, h_p2p_deps_sizes, p2p_num_nodes);
  alloc_and_copy(&d_m2l_nodes, h_m2l_nodes, m2l_num_nodes);
  alloc_and_copy(&d_m2l_deps_array, h_m2l_deps_array, m2l_deps_tot);
  alloc_and_copy(&d_m2l_deps_offsets, h_m2l_deps_offsets, m2l_num_nodes);
  alloc_and_copy(&d_m2l_deps_sizes, h_m2l_deps_sizes, m2l_num_nodes);
  timer.stop();
  printf("%-20s %12.8f\n", "  Transfer Time (s) ", timer.elapsed());

  timer.start();
  p2p_kernel<<<p2p_num_nodes, 128>>>(d_fmm, d_p2p_nodes, d_p2p_deps_array,
                                     d_p2p_deps_offsets, d_p2p_deps_sizes);
  CUDACHK(cudaGetLastError());
  CUDACHK(cudaDeviceSynchronize());
  timer.stop();
  printf("    %-16s %12.8f\n", "P2P Time (s) ", timer.elapsed());

  timer.start();
  m2l_kernel<<<m2l_num_nodes, 32>>>(d_fmm, d_m2l_nodes, d_m2l_deps_array,
                                     d_m2l_deps_offsets, d_m2l_deps_sizes);
  CUDACHK(cudaGetLastError());
  CUDACHK(cudaDeviceSynchronize());
  timer.stop();
  printf("    %-16s %12.8f\n", "M2L Time (s) ", timer.elapsed());

  free(h_p2p_nodes);
  free(h_p2p_deps_array);
  free(h_p2p_deps_offsets);
  free(h_p2p_deps_sizes);
  free(h_m2l_nodes);
  free(h_m2l_deps_array);
  free(h_m2l_deps_offsets);
  free(h_m2l_deps_sizes);

  device_free(d_p2p_nodes);
  device_free(d_p2p_deps_array);
  device_free(d_p2p_deps_offsets);
  device_free(d_p2p_deps_sizes);
  device_free(d_m2l_nodes);
  device_free(d_m2l_deps_array);
  device_free(d_m2l_deps_offsets);
  device_free(d_m2l_deps_sizes);
}

template <class T>
void perform_traversals(FMM<T>* fmm)
{
#pragma omp parallel
#pragma omp single
  printf("Running on %d threads\n", omp_get_num_threads());

  Timer timer;
  Timer tot_timer;

  timer.start();
  tot_timer.start();
#pragma omp parallel
#pragma omp single
  upwards_pass(fmm, &fmm->nodes[fmm->root]);
  timer.stop();
  printf("\n");
  printf("%-20s %12.8f\n", "Upwards Time (s) ", timer.elapsed());

  FMM<T>* h_fmm;
  FMM<T>* d_fmm;

  init_device_fmm(fmm, &h_fmm, &d_fmm);

  timer.start();
  cuda_dtt(fmm, h_fmm, d_fmm);
  timer.stop();
  printf("%-20s %12.8f\n", "DTT Time (s) ", timer.elapsed());

  fini_device_fmm(fmm, h_fmm, d_fmm);

  timer.start();
#pragma omp parallel
#pragma omp single
  downwards_pass(fmm, &fmm->nodes[fmm->root]);
  timer.stop();
  printf("%-20s %12.8f\n", "Downwards Time (s) ", timer.elapsed());

  tot_timer.stop();
  printf("--------------------\n");
  printf("%-20s %12.8f\n", "Total Time (s) ", tot_timer.elapsed());
  printf("--------------------\n\n");
}
