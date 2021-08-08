#pragma once

#include <omp.h>

#include <Kokkos_Core.hpp>
#include <gpu-utils.hh>
#include <kernels.hh>
#include <node.hh>

#include <kokkos-utils.hh>

#include <gpu-kernels-no-atomics.hh>

#ifndef KOKKOS_SCHEDULER
#define KOKKOS_SCHEDULER TaskSchedulerMultiple
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

// template <class T>
// void dual_tree(FMM<T>* fmm, node_t<T>* target, node_t<T>* source)
//{
//  T dx = source->cx - target->cx;
//  T dy = source->cy - target->cy;
//  T dz = source->cz - target->cz;
//  T r2 = dx * dx + dy * dy + dz * dz;
//  T d1 = source->rad * static_cast<T>(2.0);
//  T d2 = target->rad * static_cast<T>(2.0);
//
//  if ((d1 + d2) * (d1 + d2) < fmm->theta2 * r2) {
//    m2l(fmm, target, source);
//  }
//  else if (source->is_leaf() && target->is_leaf()) {
//    if (target == source)
//      p2p_tiled(fmm, target);
//    else
//      p2p_tiled(fmm, target, source);
//  }
//  else {
//    T target_sz = target->rad;
//    T source_sz = source->rad;
//    if (source->is_leaf() || ((target_sz >= source_sz) && !target->is_leaf()))
//    {
//      for (size_t i = 0; i < target->num_children; ++i) {
//        node_t<T>* child = &fmm->nodes[target->child[i]];
//        dual_tree(fmm, child, source);
//      }
//    }
//    else {
//      for (size_t i = 0; i < source->num_children; ++i) {
//        dual_tree(fmm, target, &fmm->nodes[source->child[i]]);
//      }
//    }
//  }
//}

namespace Kokkos {
class Cuda;
class OpenMP;
}  // namespace Kokkos

template <class Scheduler, class T>
struct dual_tree_task {
  using value_type = void;
  using future_type = Kokkos::BasicFuture<void, Scheduler>;

  FMM<T>* fmm;
  node_t<T>* target;
  node_t<T>* source;

  KOKKOS_INLINE_FUNCTION
  dual_tree_task(FMM<T>* arg_fmm, node_t<T>* arg_target, node_t<T>* arg_source)
      : fmm{arg_fmm}, target{arg_target}, source{arg_source}
  {
  }

  template <class Sched = Scheduler>
  KOKKOS_INLINE_FUNCTION typename std::enable_if<
      std::is_same<typename Sched::execution_space, Kokkos::Cuda>::value>::type
  operator()(typename Scheduler::member_type& member)
  {
    T dx = source->cx - target->cx;
    T dy = source->cy - target->cy;
    T dz = source->cz - target->cz;
    T r2 = dx * dx + dy * dy + dz * dz;
    T d1 = source->rad * static_cast<T>(2.0);
    T d2 = target->rad * static_cast<T>(2.0);

    // TODO for some reason the compiler still tries to compile this function
#ifdef __CUDACC__
    if ((d1 + d2) * (d1 + d2) < fmm->theta2 * r2) {
      if (member.team_rank() == 0) lock(&target->m2l_lock);
      member.team_barrier();
      m2l_gpu<32, 4>(fmm, target, source);
      if (member.team_rank() == 0) unlock(&target->m2l_lock);
      member.team_barrier();
    }
    else if (source->is_leaf() && target->is_leaf()) {
      if (member.team_rank() == 0) lock(&target->p2p_lock);
      member.team_barrier();
      if (target == source)
        p2p_gpu<32, 16, 4, 32, 4>(fmm, target);
      else
        p2p_gpu<32, 16, 4, 32, 4>(fmm, target, source);
      if (member.team_rank() == 0) unlock(&target->p2p_lock);
      member.team_barrier();
    }
    else {
      T target_sz = target->rad;
      T source_sz = source->rad;
      if (source->is_leaf() ||
          ((target_sz >= source_sz) && !target->is_leaf())) {
        for (size_t i = 0; i < target->num_children; ++i) {
          node_t<T>* child = &fmm->nodes[target->child[i]];
          if (target->num_points > TASK_CUTOFF) {
            if (member.team_rank() == 0)
              Kokkos::BasicFuture<void, Scheduler> f =
                  Kokkos::task_spawn(Kokkos::TaskTeam(member.scheduler()),
                                     dual_tree_task(fmm, child, source));
          }
          else {
            dual_tree_task(fmm, child, source)(member);
          }
        }
      }
      else {
        for (size_t i = 0; i < source->num_children; ++i) {
          node_t<T>* child = &fmm->nodes[source->child[i]];
          if (source->num_points > TASK_CUTOFF) {
            if (member.team_rank() == 0)
              Kokkos::BasicFuture<void, Scheduler> f =
                  Kokkos::task_spawn(Kokkos::TaskTeam(member.scheduler()),
                                     dual_tree_task(fmm, target, child));
          }
          else {
            dual_tree_task(fmm, target, child)(member);
          }
        }
      }
    }
#endif
  }

  template <class Sched = Scheduler>
  KOKKOS_INLINE_FUNCTION
      typename std::enable_if<std::is_same<typename Sched::execution_space,
                                           Kokkos::OpenMP>::value>::type
      operator()(typename Scheduler::member_type& member)
  {
    T dx = source->cx - target->cx;
    T dy = source->cy - target->cy;
    T dz = source->cz - target->cz;
    T r2 = dx * dx + dy * dy + dz * dz;
    T d1 = source->rad * static_cast<T>(2.0);
    T d2 = target->rad * static_cast<T>(2.0);

    if ((d1 + d2) * (d1 + d2) < fmm->theta2 * r2) {
      lock(&target->m2l_lock);
      m2l(fmm, target, source);
      unlock(&target->m2l_lock);
    }
    else if (source->is_leaf() && target->is_leaf()) {
      lock(&target->p2p_lock);
      if (target == source) {
        p2p_tiled(fmm, target);
      }
      else {
        p2p_tiled(fmm, target, source);
      }
      unlock(&target->p2p_lock);
    }
    else {
      T target_sz = target->rad;
      T source_sz = source->rad;
      if (source->is_leaf() ||
          ((target_sz >= source_sz) && !target->is_leaf())) {
        for (size_t i = 0; i < target->num_children; ++i) {
          node_t<T>* child = &fmm->nodes[target->child[i]];
          if (target->num_points > TASK_CUTOFF) {
            Kokkos::BasicFuture<void, Scheduler> f =
                Kokkos::task_spawn(Kokkos::TaskSingle(member.scheduler()),
                                   dual_tree_task(fmm, child, source));
          }
          else {
            dual_tree_task(fmm, child, source)(member);
          }
        }
      }
      else {
        for (size_t i = 0; i < source->num_children; ++i) {
          node_t<T>* child = &fmm->nodes[source->child[i]];
          dual_tree_task(fmm, target, child)(member);
        }
      }
    }
  }
};

template <class Scheduler, class T>
typename std::enable_if<std::is_same<typename Scheduler::execution_space,
                                     Kokkos::OpenMP>::value>::type
kokkos_dtt(FMM<T>* fmm)
{
  printf("openmp\n");
  const size_t min_block_size = 64;
  const size_t max_block_size = 1024;
  const size_t super_block_size = 4096;
  const size_t memory_capacity = 1024 * 1024 * 1024;

  Scheduler sched(typename Scheduler::memory_space(), memory_capacity,
                  min_block_size, std::min(max_block_size, memory_capacity),
                  std::min(super_block_size, memory_capacity));
  node_t<T>* root_node = fmm->nodes + fmm->root;
  Kokkos::BasicFuture<void, Scheduler> f = Kokkos::host_spawn(
      Kokkos::TaskSingle(sched),
      dual_tree_task<Scheduler, T>(fmm, root_node, root_node));
  Kokkos::wait(sched);
}

template <class Scheduler, class T>
typename std::enable_if<std::is_same<typename Scheduler::execution_space,
                                     Kokkos::Cuda>::value>::type
kokkos_dtt(FMM<T>* fmm)
{
  printf("cuda\n");
  const size_t min_block_size = 128;
  const size_t max_block_size = 1024;
  const size_t super_block_size = 4096;
  const size_t memory_capacity = 1024 * 1024 * 1024;

#ifdef __CUDACC__
  const int stack_size = 8192;
  CUDACHK(cudaDeviceSetLimit(cudaLimitStackSize, stack_size));
#endif

  Scheduler sched(typename Scheduler::memory_space(), memory_capacity,
                  min_block_size, std::min(max_block_size, memory_capacity),
                  std::min(super_block_size, memory_capacity));

  FMM<T>* d_fmm;
  FMM<T>* h_fmm;

  init_device_fmm(fmm, &h_fmm, &d_fmm);

  node_t<T>* root_node = h_fmm->nodes + h_fmm->root;

  Kokkos::BasicFuture<void, Scheduler> f = Kokkos::host_spawn(
      Kokkos::TaskTeam(sched),
      dual_tree_task<Scheduler, T>(d_fmm, root_node, root_node));
  Kokkos::wait(sched);

  fini_device_fmm(fmm, h_fmm, d_fmm);
}

template <class T>
void perform_traversals(FMM<T>* fmm)
{
  using Scheduler =
      Kokkos::KOKKOS_SCHEDULER<Kokkos::DefaultExecutionSpace>;

  Kokkos::initialize();

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
  kokkos_dtt<Scheduler>(fmm);
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


