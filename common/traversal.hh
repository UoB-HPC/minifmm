#pragma once

#include <omp.h>

#include <node.hh>
#include <kernels.hh>

template <class T>
void upwards_pass(FMM<T>* fmm, node_t<T>* node)
{
  for (size_t i = 0; i < node->num_children; ++i) {
    upwards_pass(fmm, &fmm->nodes[node->child[i]]);
  }

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
      downwards_pass(fmm, &fmm->nodes[node->child[i]]);
    }
  }
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
    m2l(fmm, target, source);
  }
  else if (source->is_leaf() && target->is_leaf()) {
    if (target == source)
      p2p_tiled(fmm, target);
    else
      p2p_tiled(fmm, target, source);
  }
  else {
    T target_sz = target->rad;
    T source_sz = source->rad;
    if (source->is_leaf() || ((target_sz >= source_sz) && !target->is_leaf())) {
      for (size_t i = 0; i < target->num_children; ++i) {
        node_t<T>* child = &fmm->nodes[target->child[i]];
        dual_tree(fmm, child, source);
      }
    }
    else {
      for (size_t i = 0; i < source->num_children; ++i) {
        dual_tree(fmm, target, &fmm->nodes[source->child[i]]);
      }
    }
  }
}

template <class T>
void perform_traversals(FMM<T>* fmm)
{
  printf("Running in serial\n");

  Timer timer;
  Timer tot_timer;

  timer.start();
  tot_timer.start();
  upwards_pass(fmm, &fmm->nodes[fmm->root]);
  timer.stop();
  printf("\n");
  printf("%-20s %12.8f\n", "Upwards Time (s) ", timer.elapsed());

  timer.start();
  dual_tree(fmm, &fmm->nodes[fmm->root], &fmm->nodes[fmm->root]);
  timer.stop();
  printf("%-20s %12.8f\n", "DTT Time (s) ", timer.elapsed());

  timer.start();
  downwards_pass(fmm, &fmm->nodes[fmm->root]);
  timer.stop();
  printf("%-20s %12.8f\n", "Downwards Time (s) ", timer.elapsed());

  tot_timer.stop();
  printf("--------------------\n");
  printf("%-20s %12.8f\n", "Total Time (s) ", tot_timer.elapsed());
  printf("--------------------\n\n");
}
