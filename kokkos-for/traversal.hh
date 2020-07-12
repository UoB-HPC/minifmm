#pragma once

#include <omp.h>

#include <Kokkos_Core.hpp>
#include <get-deps.hh>
#include <kernels.hh>
#include <node.hh>
#include <utils.hh>

#ifdef __CUDACC__
// TODO fix this (kokkos utils needs to be included before gpu kernels)
#include <kokkos-utils.hh>

#include <gpu-kernels.hh>
#endif

#ifndef KOKKOS_SCHEDULE
#define KOKKOS_SCHEDULE Dynamic
#endif

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
void kokkos_dtt(FMM<T>* fmm)
{
  FMM<T>* d_fmm;
  FMM<T>* h_fmm;

  init_device_fmm(fmm, &h_fmm, &d_fmm);

  std::vector<std::vector<size_t>> p2p_deps(fmm->num_nodes);
  std::vector<std::vector<size_t>> m2l_deps(fmm->num_nodes);

  Timer timer;
  timer.start();
  get_deps_omp(fmm, &p2p_deps, &m2l_deps);
  timer.stop();
  printf("%-20s %12.8f\n", "  Deps Time (s) ", timer.elapsed());

  timer.start();
  size_t* p2p_nodes;
  size_t* p2p_deps_array;
  size_t* p2p_deps_offsets;
  size_t* p2p_deps_sizes;
  size_t p2p_deps_tot;
  size_t p2p_num_nodes;

  size_t* m2l_nodes;
  size_t* m2l_deps_array;
  size_t* m2l_deps_offsets;
  size_t* m2l_deps_sizes;
  size_t m2l_deps_tot;
  size_t m2l_num_nodes;

  pack_deps(p2p_deps, &p2p_nodes, &p2p_deps_array, &p2p_deps_offsets,
            &p2p_deps_sizes, &p2p_deps_tot, &p2p_num_nodes);
  pack_deps(m2l_deps, &m2l_nodes, &m2l_deps_array, &m2l_deps_offsets,
            &m2l_deps_sizes, &m2l_deps_tot, &m2l_num_nodes);
  timer.stop();
  printf("%-20s %12.8f\n", "  Pack Time (s) ", timer.elapsed());

  timer.start();
  Kokkos::View<size_t*, Kokkos::HostSpace,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      h_p2p_nodes(p2p_nodes, p2p_num_nodes);
  Kokkos::View<size_t*, Kokkos::HostSpace,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      h_m2l_nodes(m2l_nodes, m2l_num_nodes);

  Kokkos::View<size_t*, Kokkos::HostSpace,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      h_p2p_deps_array(p2p_deps_array, p2p_deps_tot);
  Kokkos::View<size_t*, Kokkos::HostSpace,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      h_m2l_deps_array(m2l_deps_array, m2l_deps_tot);

  Kokkos::View<size_t*, Kokkos::HostSpace,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      h_p2p_deps_offsets(p2p_deps_offsets, p2p_num_nodes);
  Kokkos::View<size_t*, Kokkos::HostSpace,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      h_m2l_deps_offsets(m2l_deps_offsets, m2l_num_nodes);

  Kokkos::View<size_t*, Kokkos::HostSpace,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      h_p2p_deps_sizes(p2p_deps_sizes, p2p_num_nodes);
  Kokkos::View<size_t*, Kokkos::HostSpace,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      h_m2l_deps_sizes(m2l_deps_sizes, m2l_num_nodes);

  Kokkos::View<size_t*> d_p2p_nodes("d_p2p_nodes", p2p_num_nodes);
  Kokkos::View<size_t*> d_m2l_nodes("d_m2l_nodes", m2l_num_nodes);
  Kokkos::View<size_t*> d_p2p_deps_array("d_p2p_deps_array", p2p_deps_tot);
  Kokkos::View<size_t*> d_m2l_deps_array("d_m2l_deps_array", m2l_deps_tot);
  Kokkos::View<size_t*> d_p2p_deps_offsets("d_p2p_deps_offsets", p2p_num_nodes);
  Kokkos::View<size_t*> d_m2l_deps_offsets("d_m2l_deps_offsets", m2l_num_nodes);
  Kokkos::View<size_t*> d_p2p_deps_sizes("d_p2p_deps_sizes", p2p_num_nodes);
  Kokkos::View<size_t*> d_m2l_deps_sizes("d_m2l_deps_sizes", m2l_num_nodes);

  Kokkos::deep_copy(d_p2p_nodes, h_p2p_nodes);
  Kokkos::deep_copy(d_m2l_nodes, h_m2l_nodes);
  Kokkos::deep_copy(d_p2p_deps_array, h_p2p_deps_array);
  Kokkos::deep_copy(d_m2l_deps_array, h_m2l_deps_array);
  Kokkos::deep_copy(d_p2p_deps_offsets, h_p2p_deps_offsets);
  Kokkos::deep_copy(d_m2l_deps_offsets, h_m2l_deps_offsets);
  Kokkos::deep_copy(d_p2p_deps_sizes, h_p2p_deps_sizes);
  Kokkos::deep_copy(d_m2l_deps_sizes, h_m2l_deps_sizes);
  Kokkos::fence();
  timer.stop();
  printf("%-20s %12.8f\n", "  Transfer Time (s) ", timer.elapsed());

  using policy_type =
      Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace,
                         Kokkos::Schedule<Kokkos::KOKKOS_SCHEDULE>>;
  using member_type = policy_type::member_type;

#ifdef __CUDACC__
  policy_type p2p_policy = policy_type(p2p_num_nodes, 128);
  policy_type m2l_policy = policy_type(m2l_num_nodes, 32);
#else
  policy_type p2p_policy = policy_type(p2p_num_nodes, 1);
  policy_type m2l_policy = policy_type(m2l_num_nodes, 1);
#endif

  timer.start();
  Kokkos::parallel_for(
      p2p_policy, KOKKOS_LAMBDA(member_type member) {
        const int i = member.league_rank();
        node_t<T>* target = d_fmm->nodes + d_p2p_nodes[i];
        size_t p2p_size = d_p2p_deps_sizes[i];
        size_t p2p_offset = d_p2p_deps_offsets[i];
        for (size_t j = 0; j < p2p_size; ++j) {
          size_t source_idx = d_p2p_deps_array[p2p_offset + j];
          node_t<T>* source = d_fmm->nodes + source_idx;
          if (target == source) {
        // p2p_tiled(d_fmm, target);
#ifdef __CUDA_ARCH__
            p2p_gpu<128, 128, 1, 128, 1>(d_fmm, target);
#else
            p2p_tiled(d_fmm, target);
#endif
          }
          else {
        // p2p_tiled(d_fmm, target, source);
#ifdef __CUDA_ARCH__
            p2p_gpu<128, 128, 1, 128, 1>(d_fmm, target, source);
#else
            p2p_tiled(d_fmm, target, source);
#endif
          }
        }
      });
  Kokkos::fence();
  timer.stop();
  printf("%-20s %12.8f\n", "  P2P Time (s) ", timer.elapsed());

  timer.start();
  Kokkos::parallel_for(
      m2l_policy, KOKKOS_LAMBDA(member_type member) {
        const int i = member.league_rank();
        node_t<T>* target = d_fmm->nodes + d_m2l_nodes[i];
        size_t m2l_size = d_m2l_deps_sizes[i];
        size_t m2l_offset = d_m2l_deps_offsets[i];
        for (size_t j = 0; j < m2l_size; ++j) {
          size_t source_idx = d_m2l_deps_array[m2l_offset + j];
          node_t<T>* source = d_fmm->nodes + source_idx;
#ifdef __CUDA_ARCH__
          m2l_gpu<32, 1>(d_fmm, target, source);
#else
          m2l(d_fmm, target, source);
#endif
        }
      });
  Kokkos::fence();
  timer.stop();
  printf("%-20s %12.8f\n", "  M2L Time (s) ", timer.elapsed());

  free(p2p_nodes);
  free(p2p_deps_array);
  free(p2p_deps_offsets);
  free(p2p_deps_sizes);

  free(m2l_nodes);
  free(m2l_deps_array);
  free(m2l_deps_offsets);
  free(m2l_deps_sizes);

  fini_device_fmm(fmm, h_fmm, d_fmm);
}

template <class T>
void perform_traversals(FMM<T>* fmm)
{
  Kokkos::initialize();

  printf("Running in serial\n");

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

  timer.start();
  kokkos_dtt(fmm);
  // dual_tree(fmm, &fmm->nodes[fmm->root], &fmm->nodes[fmm->root]);
  timer.stop();
  printf("%-20s %12.8f\n", "DTT Time (s) ", timer.elapsed());

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

  Kokkos::finalize();
}
