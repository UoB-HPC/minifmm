#pragma once

#include <omp.h>

#include <node.hh>
#include <kernels.hh>

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
void dual_tree(FMM<T>* fmm, node_t<T>* target, node_t<T>* source)
{
  T dx = source->cx - target->cx;
  T dy = source->cy - target->cy;
  T dz = source->cz - target->cz;
  T r2 = dx * dx + dy * dy + dz * dz;
  T d1 = source->rad * static_cast<T>(2.0);
  T d2 = target->rad * static_cast<T>(2.0);

  if ((d1 + d2) * (d1 + d2) < fmm->theta2 * r2) {
    omp_set_lock(&target->m2l_lock);
    m2l(fmm, target, source);
    omp_unset_lock(&target->m2l_lock);
  }
  else if (source->is_leaf() && target->is_leaf()) {
    omp_set_lock(&target->p2p_lock);
    if (target == source)
      p2p_tiled(fmm, target);
    else
      p2p_tiled(fmm, target, source);
    omp_unset_lock(&target->p2p_lock);
  }
  else {
    T target_sz = target->rad;
    T source_sz = source->rad;
    if (source->is_leaf() || ((target_sz >= source_sz) && !target->is_leaf())) {
      for (size_t i = 0; i < target->num_children; ++i) {
        node_t<T>* child = &fmm->nodes[target->child[i]];
#pragma omp task if(target->num_points > TASK_CUTOFF)
        dual_tree(fmm, child, source);
      }
    }
    else {
      for (size_t i = 0; i < source->num_children; ++i) {
//#pragma omp task if(source->num_points > TASK_CUTOFF && SOURCE_TASK_SPAWN)
        dual_tree(fmm, target, &fmm->nodes[source->child[i]]);
      }
    }
  }
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

  timer.start();
  #pragma omp parallel
  #pragma omp single
  dual_tree(fmm, &fmm->nodes[fmm->root], &fmm->nodes[fmm->root]);
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
}
